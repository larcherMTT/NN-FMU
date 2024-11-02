<!-- TODO LIST -->
- [ ] FMU with integrator and multiple output to test framework
- [ ] check time of FMU and state reset of rnn layer
- [ ] can the rnn layer be replaced by a classical layer using tf.scan?
- [ ] make the fmu class more general
- [ ] add learnable parameters in FMU_layer
- [ ] further generalize FMU_wrap for any type of inputs, not only Real (see simulation.py of fmpy)
- [ ] Add EXPORT to ONNX with custom node (add flag to fmu_layer to use or not a dummy layer for export)
- [ ] Improve performances


# FIXES
- [ ] directional derivatives of input/output gives 0 if an integrator is on the path
- [ ] use uint32 for the state index in the FMU_layer