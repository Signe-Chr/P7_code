from calflops import calculate_flops
import Cross_validation_models as cvm

model = cvm.FilterNet_regression(9,3072)
batch_size = 1
input_shape = (batch_size, 9)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=False,
                                      output_precision=7, )
print("Regression FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))