The yaml files here are used to control the fitting targets (dimers, liquid props, etc) and the parameters to be optimised.

Files:

- data-train.yaml: Configure which properties should be fit to along with the weight they should contribute to the objective function
- params-lj.yaml: Configure which of the force field parameters should be optimised and select the initial force field to use. Note some parameter controls
 like constraining the charge of a water model are done in the fitting script directly.
- params-de.yaml: An example parameter settings file for the DEXP force field.