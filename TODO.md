<!-- TODO LIST -->
- [ ] Create a FMU with parameters as internal variables/states and check if their directional derivatives are available
- [ ] FMU with integrator and multiple output to test framework
- [ ] check time of FMU and state reset of rnn layer
- [ ] can the rnn layer be replaced by a classical layer using tf.scan?
- [ ] make the fmu class more general


# FIXES
- [ ] Fix the get_FMU_state, set_FMU_state, and free_FMU_state functions
- [ ] implement the reset function in the FMU class
- [ ] implement the reset_state function in the FMU layer
- [ ] fix the derivatives behavior in the first step