"""Fake loopback backend only; never invokes an actual LFM3 model."""
import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from lfm3_trace_inference import complete_local,ALLOWED_MODEL
from trace_memory_bridge import MemoryBridgeError
class Stub(BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.path=="/v1/models"
        data=json.dumps({"data":[{"id":ALLOWED_MODEL}]}).encode()
        self.send_response(200);self.send_header("Content-Length",str(len(data)))
        self.end_headers();self.wfile.write(data)
    def do_POST(self):
        size=int(self.headers["Content-Length"])
        packet=json.loads(self.rfile.read(size))
        assert self.path=="/v1/chat/completions"
        assert packet["model"]==ALLOWED_MODEL
        assert packet["max_tokens"]<=512
        assert len(packet["messages"])==3
        data=json.dumps({"choices":[{"message":{"content":"synthetic answer"}}]}).encode()
        self.send_response(200);self.send_header("Content-Length",str(len(data)))
        self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass
def evidence():
    return {"schema":"trace-memory.v1","source":"trace-viewer-sqlite",
        "source_generation":"legacy-unfenced","trust":"unverified-source-metadata",
        "rows":[{"rowid":3,"source_type":"opencode","trace_kind":"tool",
                 "status":"error","day_utc":"2026-10-27"}],
        "semantic_retrieval":False,"causal_relationships_verified":False,
        "model_weights_modified":False}
class LocalLFM3Test(unittest.TestCase):
    def test_real_http_roundtrip_to_fake_loopback_server(self):
        app=ThreadingHTTPServer(("127.0.0.1",0),Stub)
        th=threading.Thread(target=app.serve_forever,daemon=True);th.start()
        self.addCleanup(lambda:(app.shutdown(),app.server_close(),th.join(2)))
        url="http://127.0.0.1:"+str(app.server_port)+"/v1"
        output=complete_local(url,"Analyze categories",evidence(),enabled=True)
        self.assertEqual(output["answer"],"synthetic answer")
        self.assertEqual(output["retrieved_metadata_rows"],1)
        self.assertFalse(output["model_weights_modified"])
    def test_refuses_wrong_local_runtime_even_when_http_is_healthy(self):
        class WrongModel(Stub):
            def do_GET(self):
                data=json.dumps({"data":[{"id":"k2-horizon-0.9b"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length",str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_POST(self):
                raise AssertionError("not allowed to infer on wrong model")
        app=ThreadingHTTPServer(("127.0.0.1",0),WrongModel)
        thread=threading.Thread(target=app.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(lambda:(app.shutdown(),app.server_close(),thread.join(2)))
        with self.assertRaisesRegex(MemoryBridgeError,"not loaded locally"):
            complete_local("http://127.0.0.1:"+str(app.server_port),
                           "test",evidence(),enabled=True)

    def test_no_remote_or_silent_execution(self):
        for url in ["https://127.0.0.1:1234/v1","http://localhost:1234/v1",
                    "http://10.0.3.27:1234/v1","http://127.0.0.1:1234/admin"]:
            with self.assertRaises(MemoryBridgeError):
                complete_local(url,"prompt",evidence(),enabled=True)
        with self.assertRaises(MemoryBridgeError):
            complete_local("http://127.0.0.1:1234","prompt",evidence())
        with self.assertRaises(MemoryBridgeError):
            complete_local("http://127.0.0.1:1234","prompt",evidence(),
                           enabled=True,model="arbitrary/remote-model")
if __name__=="__main__":unittest.main()