from __future__ import annotations

import time
from typing import Annotated

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from qwen_qlora_sql_benchmark.api.generator import LazyPeftSqlGenerator, SqlGenerator
from qwen_qlora_sql_benchmark.api.sql_validation import is_select_only, parse_valid
from qwen_qlora_sql_benchmark.api.sqlite_sandbox import execute_select_query

MAX_SCHEMA_CHARS = 12000
MAX_QUESTION_CHARS = 1000


class GenerateSqlRequest(BaseModel):
    sql_schema: Annotated[str, Field(alias="schema", min_length=1, max_length=MAX_SCHEMA_CHARS)]
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]
    execute: bool = False
    setup_sql: list[str] = Field(default_factory=list, max_length=100)
    max_rows: Annotated[int, Field(ge=1, le=100)] = 20
    timeout_ms: Annotated[int, Field(ge=1, le=5000)] = 1000

    @field_validator("sql_schema", "question")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class SqlExecutionPayload(BaseModel):
    execution_valid: bool
    row_count: int
    rows: list[list[object]]
    execution_error: str | None


class GenerateSqlResponse(BaseModel):
    sql: str | None
    parse_valid: bool
    is_select_only: bool
    latency_ms: float
    error: str | None
    execution: SqlExecutionPayload | None = None


def create_app(generator: SqlGenerator | None = None) -> FastAPI:
    app = FastAPI(title="Qwen QLoRA Text-to-SQL API", version="0.1.0")
    sql_generator = generator or LazyPeftSqlGenerator()

    @app.get("/", response_class=HTMLResponse)
    def chat_console() -> str:
        return _chat_console_html()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate-sql", response_model=GenerateSqlResponse)
    def generate_sql(request: GenerateSqlRequest) -> GenerateSqlResponse:
        started = time.perf_counter()
        try:
            sql = sql_generator.generate_sql(request.sql_schema, request.question).strip()
        except Exception as exc:
            return GenerateSqlResponse(
                sql=None,
                parse_valid=False,
                is_select_only=False,
                latency_ms=_elapsed_ms(started),
                error=f"generation failed: {exc}",
            )

        valid = parse_valid(sql)
        select_only = is_select_only(sql)
        error = None
        if not valid:
            error = "generated SQL is not parseable"
        elif not select_only:
            error = "generated SQL is not a read-only SELECT query"

        execution = None
        if request.execute and valid and select_only:
            execution_result = execute_select_query(
                sql=sql,
                setup_sql=request.setup_sql,
                timeout_ms=request.timeout_ms,
                max_rows=request.max_rows,
            )
            execution = SqlExecutionPayload(
                execution_valid=execution_result.execution_valid,
                row_count=execution_result.row_count,
                rows=execution_result.rows,
                execution_error=execution_result.execution_error,
            )

        return GenerateSqlResponse(
            sql=sql,
            parse_valid=valid,
            is_select_only=select_only,
            latency_ms=_elapsed_ms(started),
            error=error,
            execution=execution,
        )

    return app


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _chat_console_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Text-to-SQL Console</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #627084;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #b45309;
      --bad: #b91c1c;
      --good: #047857;
      --code: #101820;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button, textarea, input {
      font: inherit;
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      min-height: 100vh;
    }

    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
    }

    h1, h2 {
      margin: 0;
      letter-spacing: 0;
    }

    h1 {
      font-size: 20px;
      line-height: 1.25;
    }

    h2 {
      font-size: 13px;
      line-height: 1.3;
      color: var(--muted);
      font-weight: 650;
      text-transform: uppercase;
    }

    label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    textarea {
      width: 100%;
      min-height: 148px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      color: var(--ink);
      background: #fbfdff;
      line-height: 1.45;
    }

    textarea:focus, input:focus {
      outline: 2px solid rgba(15, 118, 110, 0.22);
      border-color: var(--accent);
    }

    .field {
      display: grid;
      gap: 8px;
    }

    .compact textarea {
      min-height: 112px;
    }

    .toggle {
      display: flex;
      align-items: center;
      gap: 9px;
      color: var(--ink);
      font-size: 14px;
      font-weight: 600;
    }

    .toggle input {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }

    .meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }

    #messages {
      overflow: auto;
      padding: 22px 24px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .message {
      max-width: 900px;
      display: grid;
      gap: 8px;
    }

    .message.user {
      align-self: flex-end;
      width: min(760px, 86%);
    }

    .bubble {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 12px 14px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .user .bubble {
      border-color: rgba(15, 118, 110, 0.32);
      background: #e9f7f5;
    }

    .role {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      background: #ffffff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    .chip.good {
      border-color: rgba(4, 120, 87, 0.26);
      color: var(--good);
      background: #ecfdf5;
    }

    .chip.bad {
      border-color: rgba(185, 28, 28, 0.26);
      color: var(--bad);
      background: #fef2f2;
    }

    pre {
      margin: 0;
      padding: 12px;
      overflow: auto;
      border-radius: 8px;
      background: var(--code);
      color: #e5eef7;
      line-height: 1.45;
    }

    table {
      border-collapse: collapse;
      width: 100%;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #ffffff;
      font-size: 13px;
    }

    td {
      border: 1px solid var(--line);
      padding: 7px 8px;
      vertical-align: top;
    }

    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      padding: 16px 24px 20px;
      border-top: 1px solid var(--line);
      background: #ffffff;
    }

    #question {
      min-height: 54px;
      max-height: 180px;
    }

    button {
      align-self: end;
      min-width: 96px;
      height: 44px;
      border: 0;
      border-radius: 8px;
      color: #ffffff;
      background: var(--accent);
      font-weight: 750;
      cursor: pointer;
    }

    button:hover {
      background: var(--accent-dark);
    }

    button:disabled {
      cursor: not-allowed;
      background: #8aa6a2;
    }

    @media (max-width: 860px) {
      .shell {
        grid-template-columns: 1fr;
      }

      .sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .main {
        min-height: 640px;
      }

      .composer {
        grid-template-columns: 1fr;
      }

      button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="field">
        <h2>Text-to-SQL Console</h2>
        <div class="meta">Endpoint: /generate-sql</div>
      </div>

      <div class="field">
        <label for="schema">Schema</label>
        <textarea id="schema" spellcheck="false">CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  name TEXT,
  country TEXT,
  signup_date TEXT
);</textarea>
      </div>

      <label class="toggle" for="execute">
        <input id="execute" type="checkbox" checked>
        Execute against SQLite sandbox
      </label>

      <div class="field compact">
        <label for="setup-sql">SQLite setup SQL</label>
        <textarea id="setup-sql" spellcheck="false">CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  name TEXT,
  country TEXT,
  signup_date TEXT
);
INSERT INTO users VALUES (1, 'Alice', 'Canada', '2024-01-10');
INSERT INTO users VALUES (2, 'Bob', 'United States', '2024-02-12');</textarea>
      </div>

      <div class="meta">
        Only read-only SELECT statements are executed. Invalid or non-SELECT SQL
        is blocked before SQLite execution.
      </div>
    </aside>

    <main class="main">
      <header>
        <h1>Text-to-SQL Console</h1>
      </header>

      <section id="messages" aria-live="polite">
        <div class="message assistant">
          <div class="role">Assistant</div>
          <div class="bubble">
            Ask a database question. The model will return SQL, validation
            metadata, and optional SQLite results.
          </div>
        </div>
      </section>

      <form class="composer" id="chat-form">
        <textarea id="question" placeholder="List Canadian users" required></textarea>
        <button id="send" type="submit">Send</button>
      </form>
    </main>
  </div>

  <script>
    const form = document.querySelector("#chat-form");
    const send = document.querySelector("#send");
    const messages = document.querySelector("#messages");
    const question = document.querySelector("#question");
    const schema = document.querySelector("#schema");
    const setupSql = document.querySelector("#setup-sql");
    const execute = document.querySelector("#execute");

    function appendMessage(role, content, extras) {
      const wrapper = document.createElement("div");
      wrapper.className = `message ${role}`;

      const roleNode = document.createElement("div");
      roleNode.className = "role";
      roleNode.textContent = role === "user" ? "You" : "Assistant";
      wrapper.appendChild(roleNode);

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = content;
      wrapper.appendChild(bubble);

      if (extras) {
        wrapper.appendChild(extras);
      }

      messages.appendChild(wrapper);
      messages.scrollTop = messages.scrollHeight;
      return wrapper;
    }

    function chip(label, good) {
      const node = document.createElement("span");
      node.className = `chip ${good ? "good" : "bad"}`;
      node.textContent = label;
      return node;
    }

    function renderResponse(payload) {
      const extras = document.createElement("div");
      extras.style.display = "grid";
      extras.style.gap = "10px";

      const chips = document.createElement("div");
      chips.className = "chips";
      chips.appendChild(chip(`parse: ${payload.parse_valid}`, payload.parse_valid));
      chips.appendChild(chip(`select: ${payload.is_select_only}`, payload.is_select_only));
      chips.appendChild(chip(`latency: ${payload.latency_ms} ms`, true));
      if (payload.execution) {
        chips.appendChild(
          chip(`execution: ${payload.execution.execution_valid}`, payload.execution.execution_valid)
        );
        chips.appendChild(chip(`rows: ${payload.execution.row_count}`, true));
      }
      extras.appendChild(chips);

      if (payload.sql) {
        const sql = document.createElement("pre");
        sql.textContent = payload.sql;
        extras.appendChild(sql);
      }

      if (payload.error) {
        const error = document.createElement("div");
        error.className = "bubble";
        error.textContent = payload.error;
        extras.appendChild(error);
      }

      if (payload.execution && payload.execution.rows.length > 0) {
        const table = document.createElement("table");
        for (const row of payload.execution.rows) {
          const tr = document.createElement("tr");
          for (const value of row) {
            const td = document.createElement("td");
            td.textContent = value === null ? "NULL" : String(value);
            tr.appendChild(td);
          }
          table.appendChild(tr);
        }
        extras.appendChild(table);
      }

      if (payload.execution && payload.execution.execution_error) {
        const error = document.createElement("div");
        error.className = "bubble";
        error.textContent = payload.execution.execution_error;
        extras.appendChild(error);
      }

      appendMessage("assistant", "SQL result", extras);
    }

    function setupStatements() {
      return setupSql.value
        .split(";")
        .map((statement) => statement.trim())
        .filter(Boolean)
        .map((statement) => `${statement};`);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) {
        return;
      }

      appendMessage("user", text);
      question.value = "";
      send.disabled = true;
      send.textContent = "Running";

      try {
        const response = await fetch("/generate-sql", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            schema: schema.value,
            question: text,
            execute: execute.checked,
            setup_sql: setupStatements(),
            max_rows: 20,
            timeout_ms: 1000
          })
        });

        const payload = await response.json();
        if (!response.ok) {
          appendMessage("assistant", JSON.stringify(payload, null, 2));
        } else {
          renderResponse(payload);
        }
      } catch (error) {
        appendMessage("assistant", `Request failed: ${error}`);
      } finally {
        send.disabled = false;
        send.textContent = "Send";
        question.focus();
      }
    });
  </script>
</body>
</html>"""


app = create_app()
