#%%
def my_func(a=1, b=2, **kwargs):
    print(a)
    print(b)
    print(kwargs)

my_func(a=3)