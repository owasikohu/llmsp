-- llmsp — self-contained Neovim client config (no plugins required).
--
-- Quickest way to try it (Neovim 0.11+; ghost text needs 0.12+):
--
--   nvim -u examples/nvim/llmsp.lua /tmp/try.py
--
-- Then in insert mode:
--   * popup completion : type, or press <C-Space> to force it; <C-y> to accept.
--   * inline ghost text: appears automatically (0.12+); press <Tab> to accept.
--
-- Or source it from your own init.lua: require or `:luafile` this file.

-- Point this at the project's virtualenv so $PATH activation isn't needed.
-- (The venv's `llmsp` script has a shebang to the venv Python.)
local cmd = { "/home/owata/llmsp/.venv/bin/llmsp" }

local init_options = {
  backend = "ollama",            -- mock | ollama | deepseek | codestral | llamacpp | vllm | tgi
  model = "qwen2.5-coder:0.5b",  -- the model pulled locally; swap for a bigger one anytime
  model_family = "qwen",         -- drives FIM stop-tokens / leaked-sentinel cleanup
  -- base_url = "http://127.0.0.1:11434",
  max_tokens = 128,
  temperature = 0.1,
  debounce_ms = 200,
  request_timeout_ms = 30000,    -- generous: first call loads the model into RAM
  multiline = "auto",            -- auto | single | multi
  context = {
    max_prompt_tokens = 1536,
    cross_file = true,           -- Layer 2: recently-edited + open-file retrieval
    retrieval = "jaccard",       -- jaccard | bm25 | none
    max_snippets = 4,
    structural = true,           -- Layer 3: tree-sitter mode decision
  },
}

local function root_dir()
  local found = vim.fs.find({ ".git", "pyproject.toml" }, { upward = true })[1]
  return found and vim.fs.dirname(found) or vim.uv.cwd()
end

vim.api.nvim_create_autocmd("FileType", {
  pattern = { "python", "lua", "javascript", "typescript", "go", "rust", "c", "cpp" },
  callback = function(args)
    vim.lsp.start({
      name = "llmsp",
      cmd = cmd,
      root_dir = root_dir(),
      init_options = init_options,
    }, { bufnr = args.buf })
  end,
})

vim.api.nvim_create_autocmd("LspAttach", {
  callback = function(args)
    local client = vim.lsp.get_client_by_id(args.data.client_id)
    if not client or client.name ~= "llmsp" then
      return
    end
    local buf = args.buf

    -- Completion popup (Neovim 0.11+). autotrigger shows suggestions as you type;
    -- <C-Space> forces a request (handy since llmsp has no trigger characters).
    if vim.lsp.completion and vim.lsp.completion.enable then
      vim.lsp.completion.enable(true, args.data.client_id, buf, { autotrigger = true })
      vim.keymap.set("i", "<C-Space>", function()
        vim.lsp.completion.get()
      end, { buffer = buf, desc = "llmsp: request completion" })
    end

    -- Ghost-text inline completion (Neovim 0.12+). <Tab> accepts the suggestion.
    -- The enable() signature shifted across 0.12 pre-releases, so try the known
    -- forms defensively rather than hard-failing attach on a mismatch.
    if vim.lsp.inline_completion and vim.lsp.inline_completion.enable then
      local ic = vim.lsp.inline_completion
      if not pcall(ic.enable, true, { bufnr = buf }) then
        pcall(ic.enable, true)
      end
      vim.keymap.set("i", "<Tab>", function()
        if not (ic.get and ic.get()) then
          return "<Tab>"
        end
      end, { expr = true, buffer = buf, desc = "llmsp: accept inline completion" })
    end

    vim.notify("[llmsp] attached to buffer " .. buf, vim.log.levels.INFO)
  end,
})

-- A tiny status command to confirm the server is talking.
vim.api.nvim_create_user_command("LlmspInfo", function()
  local clients = vim.lsp.get_clients({ name = "llmsp" })
  if #clients == 0 then
    vim.notify("[llmsp] no active client", vim.log.levels.WARN)
  else
    vim.notify("[llmsp] active: cmd=" .. table.concat(clients[1].config.cmd, " "))
  end
end, {})
