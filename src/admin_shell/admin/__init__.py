"""Framework-free admin use cases and ports.

Nothing in this package imports FastAPI, Pydantic, or a database. Services
implement the ports (scanner, download runner, load state, engine status) and
wire them in their composition root; routes/ adapts the use cases to HTTP.
"""
