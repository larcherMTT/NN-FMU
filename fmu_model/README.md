<!-- - - - - - - - -
# STYLE
- - - - - - - - -->
<style>
o { color: Orange }
b { color: DodgerBlue }
r { color: Red }
</style>

# FMU exportation Guide

Brief guide to export FMUs and re-compile them. Re-compilation is necessary if the FMU is does not include the executable for the target platform.

`Simulink` and `OpenModelica` are the tools used to export FMUs.

The integration of the FMUs in a NN framework requires the information of the <o>**partial derivatives**</o> of the outputs with respect to the inputs and learnable parameters, which in this case should be treated as inputs.

> [!NOTE]
Simulink can export FMUs both for `Co-simulation` and `Model Exchange`. However, the generation of the Directional derivatives is not supported yet.

## Simulink
The model has to expose the input and outputs through the `Inport` and `Outport` blocks. Parameters have to be defined as `Simulink.Parameter` objects. And the <r>stop time has to be set to `inf`</r>.

### Export FMU
The exportation is straightforward.
-  Save -> Export as Standalone FMU

### Re-compile FMU
Follow the instructions in the file [Re-Compile_Simulink.md](./Re-Compile_Simulink.md)

## OpenModelica
To generate the directional derivatives set the following flags in the `Simulation Setup - Translation flags`:
```
-d=-DirectionalDerivatives=true
```

to include the Modelica runtime dependencies in the FMU, add the following flag to the `Simulation Setup - Translation flags`
(set it to ```all``` to include all the dependencies):
```
--fmuRuntimeDepends=modelica
```

Improve the FMU performances with the flags (see [this paper](https://2015.international.conference.modelica.org/proceedings/html/submissions/ecp15118339_FrankeWaltherWorschechBraunBachmann.pdf)):
```
+simCodeTarget=Cpp
+cseCall
```

### Re-compile FMU
Follow the instructions in the file [Re-Compile_Openmodelica.md](./Re-Compile_Openmodelica.md)