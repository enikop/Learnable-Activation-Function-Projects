# Learnable activation function NNs

## Catastrophic forgetting in Kolmogorov-Arnold networks

### Plan
Investigation of Kolmogorov-Arnold networks from the lense of catastrophic forgetting
* Basis, example in KAN paper: Simple symbolic regression, shifted Gaussian curves
* Introduce complexity: 3D symbolic regression, shifted Gaussian spikes
* Increase complexity: Complex classification task with incremental class learning (possibly also feature permutation)

### Progress
- [x] MLP for symbolic regression using incremental learning - Gaussian peaks
- [x] KAN for symbolic regression using incremental learning - Gaussian peaks
- [x] MLP for symbolic regression using incremental learning - 3D Gaussian peaks
- [x] KAN for symbolic regression using incremental learning - 3D Gaussian peaks
- [ ] MLP for multi-feature classification task using incremental learning - Wine dataset?
- [ ] KAN for multi-feature classification task using incremental learning - Wine dataset?

### General observations so far
* MLP with ReLU forgets the last iteration when trained incrementally even in the case of simple Gaussian peaks
* KAN with minimal structure (!) retains pre-trained data much better in the case of simple Gaussian peaks
    * essentially, it turns into a B-spline approximation - locality stems from this
* MLP with ReLU forgets for 3D peaks as well
* KAN seems to be just as affected by catastrophic forgetting when increasing the dimensionality of the problem
    * the simple B-splines form an intricate network where the inputs affect each other
    * this interconnection is what the network fails to represent, e.g. it overemphasizes the role of the first feature dimension

## Learnable activation functions in MLPs

### Plan
Investigation of different approximation and interpolation curves as activation functions in MLPs. \
*Will constitute the free topic midterm assignment for Geometric modelling 2024/25/II, if accepted.*
* Implement B-spline activation function
* Implement NURBS activation function
* Implement interpolation curve-based activation functions (Lagrange, Catmull-Rom and other Hermite splines etc.)
* Visualize activations
* Compare these learnable alternatives to ReLU: training time, loss, accuracy, number of parameters, minimal network structure needed
* Compare these learnable alternatives to each other: training time, loss, accuracy, number of parameters, minimal network structure needed

## Progress
- [x] B-spline activation layer
- [x] NURBS activation layer
- [x] Lagrange activation layer
- [ ] Catmull-Rom activation layer
- [ ] Hermite spline activation layer with different tangent calculation methods
- [x] Visualizations
- [x] Simple example approximating a sine function, without weights in linear layers
- [ ] Complex experiment (on MNIST?) with measurements recorded for all learnable activations + ReLU