"""Run the Py version's comparison tool (gpu_cht_solver_r1/verify/compare_of.py, mounted read-only at /py)."""
import runpy
import sys

sys.path.insert(0, "/py")
sys.argv = ["/py/verify/compare_of.py"] + sys.argv[1:]
runpy.run_path("/py/verify/compare_of.py", run_name="__main__")
