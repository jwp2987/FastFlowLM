<p align="center">
  <a href="https://www.fastflowlm.com" target="_blank">
    <img src="assets/logo.png" alt="FastFlowLM Logo" width="200"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NPU-Optimized-red" />
</p>

## ⚡ FastFlowLM (FLM) — Unlock Ryzen™ AI NPUs

Run large language models — now with **Vision**, **Audio**, **Embedding** and **MoE** support — on **AMD Ryzen™ AI NPUs** in minutes.  
**No GPU required. Faster and over 10× more power-efficient. Supports context lengths up to 256k tokens. Ultra-Lightweight (17 MB). Installs within 20 seconds.**

📦 **The only out-of-box, NPU-first runtime built exclusively for Ryzen™ AI.**  
🤝 **Think Ollama — but deeply optimized for NPUs.**  
✨ **From Idle Silicon to Instant Power — FastFlowLM Makes Ryzen™ AI Shine.**

> FastFlowLM (FLM) supports all Ryzen™ AI Series chips with XDNA2 NPUs (Strix, Strix Halo, Kraken, and Gorgon Point).

---

## 🧪 About this fork

This is a personal fork — largely me playing around with the host-side runtime
on a Ryzen™ AI NPU, not an official release. Everything here is on the C++ host
layer; the prebuilt NPU kernels/firmware ship as binaries and are **untouched**.
Nothing below has been upstreamed. In rough order of how much I poked at it:

- **gpt-oss OpenAI tool/function calling, end-to-end.** `tools` were silently
  ignored (plain-text replies, never a tool call). Fixed prompt injection of the
  Harmony `functions` namespace, stopping generation on `<|call|>`, and parsing
  the commentary tool channel for both streaming and non-streaming. Added
  `tool_choice` support — `none`, forced-function, and `required` via constrained
  decoding that forces the Harmony commentary channel — and gave every tool call
  a unique id (several paths were emitting duplicates or the literal
  `"generate_id()"`).
- **Concurrency & memory safety.** Fixed glibc heap corruption under concurrent
  `/v1/chat/completions` bursts (#608) by holding the NPU lock until the handler
  fully returns instead of releasing it as a side effect of `send_response`.
  Added a missing virtual destructor to `npu_cmd` (sized-delete UB / heap
  corruption on load), serialized `header_print` so concurrent I/O threads stop
  interleaving log lines, and removed the undocumented 333 ms inter-request
  cooldown that was masking the #608 race (~35% faster on 5 concurrent requests).
- **KV-cache prefix reuse.** Reuse a shared prompt prefix instead of clearing and
  re-prefilling the whole context on every divergent request (55–82% fewer
  prefill tokens on the workloads measured). Because `set_context_length()` is
  unsound on some engines (gpt-oss accepts it and then returns garbage), support
  is now measured once at load time by comparing logits; `FLM_SKIP_KV_PROBE=1`
  skips it. Measured engine KV semantics are written up in
  [`docs/kv_primitives.md`](docs/kv_primitives.md).
- **Better error/diagnostic reporting.** Streaming failures are emitted as proper
  SSE frames (`context_length_exceeded`, etc.) instead of an unparseable bare
  JSON body mid-stream, and prompt-cache misses now report *why* they missed
  (which round/message diverged) instead of a bare "cache miss".
- **Host-side perf cleanups.** Actually enable the AVX2 sampler path (the
  `USEAVX2` guard was never defined for the shipped binary), a bounded-heap top-k,
  and a few tokenizer/whisper allocation fixes — all bit-identical, host-only, and
  below the noise floor end-to-end since the NPU forward pass dominates.

I also chased an NPU **firmware hang** on gpt-oss prefill (the control processor
wedges at a fixed PC during a prefill GEMM and the driver TDR-timeouts it,
surfacing as `qds_device::wait() unexpected command state`). It reproduces on
stock upstream too and lives below the host layer in the binary firmware/xclbin,
so it isn't fixable in this tree — it's documented, not patched.

---

## 🔗 Quick Links

  🔽 **[Download](https://github.com/FastFlowLM/FastFlowLM/releases/latest/download/flm-setup.exe)** | 📊 **[Benchmarks](https://fastflowlm.com/docs/benchmarks/)** | 📦 **[Model List](https://fastflowlm.com/docs/models/)**  

  🐧 **[Linux Getting Started Guide](./docs/linux-getting-started.md)**

  📖 **[Docs](https://fastflowlm.com/docs)** | 📺 **[Demos](https://www.youtube.com/playlist?list=PLf87s9UUZrJoDdz639Yc6w1UTyJ4cFHZ1)** | 💬 **[Discord](https://discord.gg/z24t23HsHF)** 

---

## 🚀 Quick Start

A packaged FLM Windows installer is available here: [**flm-setup.exe**](https://github.com/FastFlowLM/FastFlowLM/releases/latest/download/flm-setup.exe). For more details, see the [release notes](https://github.com/FastFlowLM/FastFlowLM/releases/).

📺 [**Watch the quick start video (Windows)**](https://www.youtube.com/watch?v=mYOfDNkyBII)

> [!IMPORTANT]  
> ⚠️ Ensure NPU driver version is **>= 32.0.203.304** (`.304` is the minimum requirement but `.311` is recommended; check via Task Manager→Performance→NPU or Device Manager).  
> ⚙️ **Tip:**
>   * **RECOMMENDED**: Try running **Windows Update** or **[Driver Download](https://www.amd.com/en/support)**.
>   * **[Official AMD Install Doc](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)** *(AMD account required)*.
>   * **[Unofficial forum downloads](https://www.elevenforum.com/t/drivers-amd-npu-ryzen-8xxx-9xxx-apu.24220/)** *(CAUTION, we do not hold responsible for what you download here)*.

After installation, open **PowerShell** (`Win + X → I`). To run a model in terminal (**CLI Mode**):
```powershell
flm run llama3.2:1b
```
> **Notes:**
> - Internet access to HuggingFace is required to download the optimized model kernels.
> - Sometimes downloads from HuggingFace may get corrupted. If this happens, run `flm pull <model_tag> --force` (e.g. `flm pull llama3.2:1b --force`) to re-download and fix them.
> - By default, models are stored in:
>   - **Windows**: `C:\Users\<USER>\Documents\flm\models\`
>   - **Linux**: `~/.config/flm/`
> - During installation on Windows, you can select a different base folder (e.g., if you choose `C:\Users\<USER>\flm`, models will be saved under `C:\Users\<USER>\flm\models\`).
> - On Linux, you can override the default location by setting the `FLM_MODEL_PATH` environment variable.
> - To disable the startup version check, set `FLM_DISABLE_UPDATE_CHECK=1`.
> - To skip the load-time KV-truncation probe, set `FLM_SKIP_KV_PROBE=1`. The engine is then treated as not supporting truncation, so divergent prompts are prefilled in full and `set_context_length()` is never called. This is a diagnostic escape hatch for isolating NPU/engine faults; see [`docs/kv_primitives.md`](docs/kv_primitives.md).
> - ⚠️ If HuggingFace is not accessible in your region, manually download the model ([check this issue](https://github.com/FastFlowLM/FastFlowLM/issues/2)) and place it in the chosen directory.   

🎉🚀 FastFlowLM (FLM) is ready — your NPU is unlocked and you can start chatting with models right away!

Open **Task Manager** (`Ctrl + Shift + Esc`). Go to the **Performance** tab → click **NPU** to monitor usage.  

> **⚡ Quick Tips:**  
> - Use `/verbose` during a session to turn on performance reporting (toggle off with `/verbose` again).   
> - Type `/bye` to exit a conversation.  
> - Run `flm list` in PowerShell to show all available models.  

To start the local server (**Server Mode**):
```powershell
flm serve llama3.2:1b
```
> The model tag (e.g., `llama3.2:1b`) sets the initial model, which is optional. If another model is requested, FastFlowLM will automatically switch to it. Local server is on port 52625 (default).  

**[![FastFlowLM Docs](https://img.shields.io/badge/FastFlowLM-Detailed%20Instructions-red?style=flat&logo=readthedocs)](https://fastflowlm.com/docs/instructions/)**

---

## 📰 In the News

- 07/17/2026 🎉 FLM is now part of AMD **[news](https://www.amd.com/en/blogs/2026/fastflowlm-joins-amd-to-advance-ai-inference.html)**.

- 03/11/2026 🎉 FLM now supports Linux 🐧 ! To get started, check out the **[quick start guide](https://fastflowlm.com/docs/install_lin/)** or the **[Lemonade Server docs](https://lemonade-server.ai/flm_npu_linux.html)**, and watch the **[short video](https://www.youtube.com/watch?v=tXRchP3sKA8)** for a quick walkthrough of FLM on Linux via Lemonade 🍋.

- 10/01/2025 🎉 FLM was integrated into AMD's **[Lemonade Server](https://lemonade-server.ai/)** 🍋. Watch this **[short demo](https://www.youtube.com/watch?v=w0Tb3h4WUnE)** about using FLM in Lemonade.

---

## 🧠 Local AI on NPU

FLM makes it easy to run cutting-edge **LLMs** (and now **VLMs**) locally with:
- ⚡ Fast and low power
- 🧰 Simple CLI and API (REST and OpenAI API)
- 🔐 Fully private and offline

No model rewrites, no tuning — it just works.

---

## ✅ Highlights

- **Runs fully on AMD Ryzen™ AI NPU** — no GPU or CPU load
- **Lightweight runtime (17 MB)** — installs within **20 seconds**, easy to integrate    
- **Developer-first flow** — like Ollama, but optimized for NPU  
- **Support for long context windows** — up to 256k tokens (e.g., Qwen3-4B-Thinking-2507)  
- **No low-level tuning required** — You focus on your app, we handle the rest

---

## 📄 License

- All orchestration code and CLI tools are open-source under the [MIT License](./LICENSE_RUNTIME.txt).  
- These NPU-accelerated binary kernels are completely free for any use, including commercial use.
- Please acknowledge FastFlowLM in your README/project page (or product) as follows:
  ```
  Powered by [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM)
  ```
  
---

💬 Have **feedback/issues** or want **early access** to our new releases? [Open an issue](https://github.com/fastflowlm/fastflowlm/issues/new) or [Join our Discord community](https://discord.gg/z24t23HsHF)

---

## 🙏 Acknowledgements

- Powered by the advanced **AMD Ryzen™ AI NPU architecture**
- Inspired by the widely adopted [llama.cpp](https://github.com/ggml-org/llama.cpp) and [Ollama](https://github.com/ollama/ollama)
- Tokenization accelerated with [MLC-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
- Chat formatting via [Google/minja](https://github.com/google/minja)
- Low-level kernels optimized using the powerful [IRON](https://github.com/amd/iron)+[AIE-MLIR](https://github.com/Xilinx/mlir-aie)

---

## 🛠️ Building from Source

For developers who want to build FastFlowLM from source, we provide CMake presets for a convenient and consistent build experience.

### Prerequisites

- Git
- CMake (version 3.22 or higher)
- A C++20 compatible compiler (e.g., GCC, Clang, MSVC)
- Ninja (recommended)

### Build Instructions

More details on the exact procedure, with dependencies to be installed, for linux can be found in [linux-getting-started.md](docs/linux-getting-started.md).

1.  **Clone the repository:**

    ```bash
    git clone --recursive https://github.com/FastFlowLM/FastFlowLM.git
    cd FastFlowLM/src
    ```

2.  **Configure CMake using presets:**

    -   **For Linux:**

        ```bash
        cmake --preset linux-default
        ```

        This will configure the build to install to `/opt/fastflowlm`.

    -   **For Windows (in a developer command prompt):**

        ```bash
        cmake --preset windows-default
        ```

3.  **Build the project:**

    ```bash
    cmake --build build
    ```

4.  **Install the project (optional):**

    -   **For Linux:**

        ```bash
        sudo cmake --install build
        ```

    -   **For Windows (with administrator privileges):**

        ```bash
        cmake --install build
        ```
