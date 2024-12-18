<!-- TODO LIST -->
- [ ] add learnable parameters in FMU_layer
- [ ] further generalize FMU_wrap for any type of inputs, not only Real (see simulation.py of fmpy)
- [ ] Add EXPORT to ONNX with custom node (add flag to fmu_layer to use or not a dummy layer for export) (see GPT)
- [ ] Improve performances
- [ ] Organize folders with examples
- [ ] Add example with modelica code generation from Maple or sympy
- [ ] Introduce and test feedback loops (see https://github.com/csirmaz/superloop)
- [ ] Get layer definition from DT (with learnable parameters)
- [ ] Check if learnable parameters can be treated as (causality = "parameter" and variability = "tunable") and chek if the directional derivatives are computed correctly -> causality seems to not be supported by omc
- [ ] Create maple toolbox for modelica code generation
  - [ ] add handling of learnable parameters
  - [ ] add `reinit` functionality for states
- [ ] update readme with exportation with cvode integrator and with make install command (probably also .fmu exportation, see cmakelists.txt)



# FIXES
- [ ] directional derivatives of input/output gives 0 if an integrator is on the path (it may be correct since in order to se a change on the output we need a time variation (it changes the slope of the output and not the output directly))
- [ ] use uint32 for the state index in the FMU_layer