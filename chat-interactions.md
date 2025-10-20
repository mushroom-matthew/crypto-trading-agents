Totally—since you’ve dockerized the stack, the easiest way to “hit a tool” without an MCP-aware client is to expose a couple of thin HTTP shims in your mcp_server/app.py and call them with curl. Your server is already up on 0.0.0.0:8080, so we just need routes.

0) (Optional) add a simple /healthz

You saw Not Found earlier because there isn’t a health route. Add this near the bottom of app.py:

@app.custom_route("/healthz", methods=["GET"])
async def healthz(_request):
    return JSONResponse({"ok": True})

1) Add HTTP shims for a few tools

Drop this block into app.py (just above the if __name__ == "__main__":):

# ---- Simple HTTP shims to invoke selected MCP tools ----
@app.custom_route("/tools/start_market_stream", methods=["POST"])
async def http_start_market_stream(request: Request) -> Response:
    body = await request.json()
    symbols = body.get("symbols") or []
    interval_sec = int(body.get("interval_sec", 1))
    load_historical = bool(body.get("load_historical", True))
    result = await start_market_stream(symbols, interval_sec, load_historical)
    return JSONResponse(result)

@app.custom_route("/tools/get_portfolio_status", methods=["GET", "POST"])
async def http_get_portfolio_status(_request: Request) -> Response:
    result = await get_portfolio_status()
    return JSONResponse(result)

@app.custom_route("/tools/place_mock_order", methods=["POST"])
async def http_place_mock_order(request: Request) -> Response:
    body = await request.json()
    orders = body.get("orders") or []
    # let pydantic/dataclass conversion inside tool handle types
    result = await place_mock_order(orders)
    return JSONResponse(result)

@app.custom_route("/tools/get_transaction_history", methods=["GET"])
async def http_get_tx(_request: Request) -> Response:
    result = await get_transaction_history()
    return JSONResponse(result)

@app.custom_route("/tools/trigger_evaluation", methods=["POST"])
async def http_trigger_eval(request: Request) -> Response:
    body = await request.json()
    window_days = int(body.get("window_days", 7))
    force = bool(body.get("force", False))
    result = await trigger_performance_evaluation(window_days, force)
    return JSONResponse(result)


Then rebuild/restart the app service:

docker compose up -d --build app
docker compose logs -f app


(Keep the task queue name consistent with your worker—either change the worker to mcp-tools or change the task_queue="mcp-tools" calls in app.py to crypto-agents.)

2) Run the “demo” flow via curl
A) Start the market stream (+ optional historical load)
curl -s http://localhost:8080/tools/start_market_stream \
  -H "Content-Type: application/json" \
  -d '{"symbols":["BTC/USD","ETH/USD","SOL/USD"],"interval_sec":1,"load_historical":true}' | jq


You should see { "workflow_id": "...", "run_id": "..." }.
(Temporal logs will show tasks on your chosen task queue.)

B) (Optional) Watch tick signals SSE

Your server already has a signal endpoint. Open a second terminal:

curl -N http://localhost:8080/signal/market_tick?after=0


You’ll see streaming data: {...} lines as ticks arrive.

C) Check portfolio status
curl -s http://localhost:8080/tools/get_portfolio_status | jq

D) Place a mock order (atomic batch supported)
curl -s http://localhost:8080/tools/place_mock_order \
  -H "Content-Type: application/json" \
  -d '{
    "orders": [
      {"symbol":"BTC/USD","side":"BUY","qty":0.001,"price":50000,"type":"market"},
      {"symbol":"ETH/USD","side":"SELL","qty":0.05,"price":3000,"type":"market"}
    ]
  }' | jq

E) See transaction history
curl -s http://localhost:8080/tools/get_transaction_history | jq

F) Trigger the judge evaluation
curl -s http://localhost:8080/tools/trigger_evaluation \
  -H "Content-Type: application/json" \
  -d '{"window_days":7,"force":true}' | jq

3) Quick sanity: Temporal + queues

List namespaces (works already for you):

docker compose exec temporal tctl --address temporal:7233 namespace list


Confirm your worker is polling the same queue you’re using in app.py. If you switch to mcp-tools, update worker config/env to:

services:
  worker:
    environment:
      - TASK_QUEUE=mcp-tools
    depends_on:
      temporal:
        condition: service_healthy


(With a Temporal healthcheck as shown earlier.)

Extras you already have

Signals API (present):

POST a synthetic tick:

curl -s -X POST http://localhost:8080/signal/market_tick \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC/USD","price":50123.45,"timestamp":'"$(date +%s)"'}' -i


Stream them (SSE) as shown above.

Workflow status (present):

curl -s http://localhost:8080/workflow/<workflow_id>/<run_id> | jq


That’s it. With those tiny HTTP shims, you can replicate the README demo flow from plain curl inside your Docker setup—no MCP client required. If you later want a proper MCP client experience, point an MCP-aware tool (e.g., Claude Desktop, MCP Inspector) at http://localhost:8080 and all @app.tool methods will show up automatically.