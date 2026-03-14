# src/kubeflow_mcp/cli.py
import argparse
from kubeflow_mcp.server import create_server

def main():
    parser = argparse.ArgumentParser(description="Kubeflow MCP Server")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--clients", default="trainer")
    serve.add_argument("--persona", default="ml-engineer")
    serve.add_argument("--transport", default="stdio",
                       choices=["stdio", "streamable-http"])

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        persona = getattr(args, 'persona', 'ml-engineer')
        server = create_server(persona=persona)
        server.run()

if __name__ == "__main__":
    main()