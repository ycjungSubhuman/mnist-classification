import torch

def require_eq(a, b):
    if a != b:
        raise RuntimeError('require_eq(a, b) failed: a={} b={}'
                           .format(a, b))

'''
if __debug__ is on, given expect_in, assert output of mod is of expect_out
'''
def check_in_out_dim(mod, expect_in, expect_out):
    if __debug__:
        in_data = torch.zeros(10,*expect_in)
        out_data = mod(in_data)
        # Batch size should be preserved
        require_eq(in_data.size()[0], out_data.size()[0])
        # out_data should have dim+1 of expect_out
        require_eq(out_data.dim(), len(expect_out)+1)
        # out_data should have size of expect_out
        require_eq(list(out_data.size())[1:], expect_out)
        
