# def numerical_gradients(model, input_data1, input_data2, output_labels, epsilon=1e-5):
#     weight_gradients = []
#     bias_gradients = []
#     for i in range(len(model.weight)):
#         tempmodel_weight_plus= model
#         tempmodel_weight_plus.weight[i] += epsilon
#         tempmodel_bias_plus= model
#         tempmodel_bias_plus.bias[i] += epsilon
#         tempmodel_weight_minus= model
#         tempmodel_weight_minus.weight[i] -= epsilon
#         tempmodel_bias_minus= model
#         tempmodel_bias_minus.bias[i] -= epsilon

#         tempmodel_weight_plus_prediction= tempmodel_weight_plus.forward(input_data1, input_data2)
#         tempmodel_weight_minus_prediction= tempmodel_weight_minus.forward(input_data1, input_data2)
#         tempmodel_bias_plus_prediction= tempmodel_bias_plus.forward(input_data1, input_data2)
#         tempmodel_bias_minus_prediction= tempmodel_bias_minus.forward(input_data1, input_data2)
#         brt= torch.nn.MSELoss()
#         weight_loss_plus= brt(tempmodel_weight_plus_prediction, output_labels)
#         weight_loss_minus= brt(tempmodel_weight_minus_prediction, output_labels)
#         bias_loss_plus= brt(tempmodel_bias_plus_prediction, output_labels)
#         bias_loss_minus= brt(tempmodel_bias_minus_prediction, output_labels)

#         weight_numerical_gradient = (weight_loss_plus - weight_loss_minus) / (2 * epsilon)
#         bias_numerical_gradient = (bias_loss_plus - bias_loss_minus) / (2 * epsilon)
#         weight_gradients.append(weight_numerical_gradient)
#         bias_gradients.append(bias_numerical_gradient)
    
#     return weight_gradients, bias_gradients

# def gradient_check(model, input_data, output_labels):
#     predictions= model.forward(input_data)
#     analytical_weight_gradients, analytical_bias_gradients = model.compute_gradients(predictions, output_labels, mse_grad)
#     weight_numerical_gradients, bias_numerical_gradients = numerical_gradients(model, input_data, output_labels)

#     for i in range(len(model.weight)):
#         weight_analytical_gradient = analytical_weight_gradients[i]
#         weight_numerical_gradient = weight_numerical_gradients[i]
#         bias_analytical_gradient = analytical_bias_gradients[i]
#         bias_numerical_gradient = bias_numerical_gradients[i]  

#         weight_is_close = abs(weight_analytical_gradient - weight_numerical_gradient) < 1e-7
#         if not weight_is_close:
#             print(f"Gradient check failed for weight {i}")
#             print(f"Analytical gradient: {weight_analytical_gradient}")
#             print(f"Numerical gradient: {weight_numerical_gradient}")
#         else:
#             print(f"Gradient check passed for parameter bias")

#         bias_is_close = abs(bias_analytical_gradient - bias_numerical_gradient) < 1e-7
#         if not bias_is_close:
#             print(f"Gradient check failed for bias {i}")
#             print(f"Analytical gradient: {bias_analytical_gradient}")
#             print(f"Numerical gradient: {bias_numerical_gradient}")
#         else:
#             print(f"Gradient check passed for parameter weight")


# gradient_check(model, inputs, inputs1, output_labels)
# def LayerNormalizationDerivative2(InputData: torch.Tensor, epsilon= 0.0000000000001):
#     mean = InputData.mean(-1, keepdim = True)
#     std = InputData.std(-1, keepdim = True, unbiased=False)
#     n= InputData.shape[1]
#     a= ((std+epsilon)*(1-1/n))
#     b= (InputData-mean)
#     c= (torch.div(b,n*(std+epsilon)))
#     top= a- (b*c)
#     bottom= c**2
#     derivative= torch.div(top, bottom)
#     return derivative
# def LayerNormalizationDerivative(InputData: torch.Tensor, epsilon= 0.0000000000001):
#     mean = InputData.mean(-1, keepdim = True)
#     std = InputData.std(-1, keepdim = True, unbiased=False)
#     n= InputData.shape[1]
#     derivative= torch.div(n-1, n*(std+epsilon))- torch.div((InputData-mean)**2, n*(std**3 +epsilon))
#     return derivative