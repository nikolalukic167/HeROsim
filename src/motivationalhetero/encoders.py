PLATFORM_TYPES = ['rpiCpu', 'xavierCpu', 'xavierGpu', 'xavierDla', 'pynqFpga']

def get_platform_type_encoder():
    from sklearn.preprocessing import OneHotEncoder



    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit([[device_type] for device_type in PLATFORM_TYPES])
    return encoder

def get_device_type_encoder():
    from sklearn.preprocessing import OneHotEncoder

    device_types = ['xavier', 'rpi', 'pyngFpga']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit([[device_type] for device_type in device_types])
    return encoder