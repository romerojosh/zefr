import zefr

inputfile = "input_vortex"

ZEFR = zefr.zefr()
ZEFR.initialize(inputfile)
ZEFR.setup_solver()

for i in range(1,1000):
    ZEFR.do_step()
    if i%100 == 0:
        ZEFR.write_residual()
    if i%200 == 0:
        ZEFR.write_solution()

ZEFR.write_residual()
ZEFR.write_solution()

print "Run complete!"
