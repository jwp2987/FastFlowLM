#pragma once
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <locale>
#include <random>
#include <utility>
#include <vector>
#include <algorithm>
#include "nlohmann/json.hpp"
#include "AutoModel/automodel.hpp"

using json = nlohmann::ordered_json;

struct cache_match_info_t {
    bool usable = false;         // whether the cache can be reused
    size_t matched_rounds = 0;   // number of leading rounds whose checksum matched
    size_t cached_rounds = 0;    // number of rounds currently held in the cache
    size_t total_rounds = 0;     // number of rounds in the incoming request
    bool tools_matched = false;  // whether the tool definitions matched

    // Diagnostics for the first message that failed to match. A prefix cache is
    // useless if any earlier message is rewritten between turns, so when a match
    // fails it matters *which* message changed and what it now looks like.
    std::string divergent_role;      // role of the first non-matching message
    size_t divergent_len = 0;        // its content length in the new request
    std::string divergent_head;      // leading characters of its content
    std::string divergent_tail;      // trailing characters of its content
};

class PromptCache {
private:
    std::vector<uint64_t> message_checksums_;
    std::vector<uint64_t> tool_checksums_;

    uint64_t _calculate_single_message_checksum(const json& message) {
        // Hash only the content-bearing fields so messages produced by
        // different backends (e.g. local vs cloud) compare equal as long
        // as their semantic payload matches.
        uint64_t sum = FNV_OFFSET_BASIS;
        auto mix = [&](const char* key) {
            if (message.contains(key)) {
                const std::string s = message[key].dump();
                sum = _calculate_checksum(s.data(), s.size(), sum);
            }
        };
        mix("role");
        mix("content");
        mix("tool_calls");
        mix("tool_call_id");
        mix("name");
        return sum;
    }

    std::vector<uint64_t> _calculate_message_checksums(const json& messages, size_t end) {
        const size_t message_count = std::min(end, messages.size());
        std::vector<uint64_t> checksums;
        checksums.reserve(message_count);
        for (size_t i = 0; i < message_count; ++i) {
            checksums.push_back(_calculate_single_message_checksum(messages[i]));
        }
        return checksums;
    }

    std::vector<uint64_t> _calculate_tool_checksums(const json& tools) {
        std::vector<uint64_t> tool_checksums;
        tool_checksums.reserve(tools.size());
        for (const auto& tool : tools) {
            const std::string tool_string = tool.dump();
            tool_checksums.push_back(_calculate_checksum(tool_string.data(), tool_string.size()));
        }
        return tool_checksums;
    }

    // FNV-1a. The previous additive word sum was order-independent, so any
    // reordering of 8-byte-aligned words collided (e.g. "AAAAAAAABBBBBBBB" and
    // "BBBBBBBBAAAAAAAA" hashed equal). It also read the payload through an
    // unaligned uint64_t*. Byte-at-a-time keeps this position-sensitive.
    static constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME = 1099511628211ULL;

    uint64_t _calculate_checksum(const void* p, size_t len, uint64_t sum = FNV_OFFSET_BASIS) {
        const uint8_t* data = static_cast<const uint8_t*>(p);
        uint64_t hash = sum;
        for (size_t i = 0; i < len; ++i) {
            hash ^= static_cast<uint64_t>(data[i]);
            hash *= FNV_PRIME;
        }
        return hash;
    }
    
public:
    PromptCache() : message_checksums_(), tool_checksums_() {}

    bool can_use_tool_cache(json& tools) {
        std::vector<uint64_t> new_tool_checksums = _calculate_tool_checksums(tools);

        if (tool_checksums_.size() == new_tool_checksums.size()){
            for (size_t i = 0; i < tool_checksums_.size(); ++i) {
                if (tool_checksums_[i] != new_tool_checksums[i]) {
                    tool_checksums_ = std::move(new_tool_checksums);
                    return false;
                }
            }
            return true;
        }
        else {
            tool_checksums_ = std::move(new_tool_checksums);
            return false;
        }
    }

    void reset_tool_checksum() {
        tool_checksums_.clear();
    }

    void update_tool_checksum(json& tools) {
        if (!tools.is_array() || tools.empty()) {
            reset_tool_checksum();
            return;
        }

        tool_checksums_ = _calculate_tool_checksums(tools);
    }


    bool can_use_message_cache(json& messages, chat_template_type_t template_type) {
        (void)template_type;
        if (messages.size() <= 2) {
            return false;
        }

        std::vector<uint64_t> new_checksums = _calculate_message_checksums(messages, messages.size());
        const size_t prefix_len = messages.size() - 2;
        const bool prefix_match =
            message_checksums_.size() <= prefix_len &&
            std::equal(message_checksums_.begin(), message_checksums_.end(), new_checksums.begin());
        message_checksums_ = std::move(new_checksums);
        return prefix_match;
    }

    void update_message_checksum(json& messages) {
        message_checksums_ = _calculate_message_checksums(messages, messages.size());
    }


    bool can_use_cache(json& messages, chat_template_type_t template_type, json& tools) {
        cache_match_info_t info;
        return can_use_cache(messages, template_type, tools, info);
    }

    bool can_use_cache(json& messages, chat_template_type_t template_type, json& tools, cache_match_info_t& info) {
        (void)template_type;
        info = cache_match_info_t{};
        info.total_rounds = messages.size();
        info.cached_rounds = message_checksums_.size();

        if (messages.size() <= 2) {
            update_message_checksum(messages);
            update_tool_checksum(tools);
            return false;
        }

        std::vector<uint64_t> new_checksums = _calculate_message_checksums(messages, messages.size());
        std::vector<uint64_t> new_tool_checksums = _calculate_tool_checksums(tools);

        // Count how many leading rounds of the cached checksums still match the
        // incoming request. This is the reusable KV-cache prefix length.
        size_t matched = 0;
        const size_t compare_len = std::min(message_checksums_.size(), new_checksums.size());
        while (matched < compare_len && message_checksums_[matched] == new_checksums[matched]) {
            ++matched;
        }
        info.matched_rounds = matched;

        // Capture what the first divergent message looks like now. Its head shows
        // whether the message identity changed; its tail shows whether only a
        // trailing block (e.g. an env/context footer) was regenerated.
        if (matched < messages.size()) {
            const auto& bad = messages[matched];
            info.divergent_role = bad.value("role", "?");
            std::string content;
            if (bad.contains("content") && bad["content"].is_string()) {
                content = bad["content"].get<std::string>();
            }
            else if (bad.contains("content")) {
                content = bad["content"].dump();
            }
            info.divergent_len = content.size();
            const size_t edge = 70;
            info.divergent_head = content.substr(0, std::min(edge, content.size()));
            info.divergent_tail = content.size() > edge
                ? content.substr(content.size() - edge)
                : std::string();
            // Keep the log on one line.
            for (std::string* s : {&info.divergent_head, &info.divergent_tail}) {
                for (char& c : *s) { if (c == '\n' || c == '\r' || c == '\t') c = ' '; }
            }
        }

        // Cache is reusable when every previously seen message still appears
        // (in order) at the start of the new conversation, allowing rounds
        // produced by other backends (cloud) to be appended without
        // invalidating the locally-built KV cache prefix.
        const size_t prefix_len = messages.size() - 2;
        const bool can_use_message =
            message_checksums_.size() <= prefix_len &&
            matched == message_checksums_.size();
        const bool can_use_tools = tool_checksums_ == new_tool_checksums;
        info.tools_matched = can_use_tools;

        message_checksums_ = std::move(new_checksums);
        tool_checksums_ = std::move(new_tool_checksums);

        info.usable = can_use_message && can_use_tools;
        return info.usable;
    }

    /// @brief Reset the checksum to force cache miss
    /// @note Clears the stored checksums so the next can_use_cache call misses.
    void reset() {
        message_checksums_.clear();
        reset_tool_checksum();
    }
};