from mpi4py import MPI
import zefr

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

inputfile = "input_channel"
nGrids = 1
gridID = 0

zefr.initialize(comm, "input_channel")
ZEFR = zefr.get_zefr_object()
ZEFR.read_input(inputfile)
ZEFR.setup_solver()

for i in range(1,1000):
    ZEFR.do_step()
    if i%100 == 0:
        ZEFR.write_residual()
    if i%200 == 0:
        ZEFR.write_solution()

ZEFR.write_residual()
ZEFR.write_solution()

if rank==0:
    print "Run complete!"
