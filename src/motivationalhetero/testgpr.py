from src.motivationalhetero.encoders import get_platform_type_encoder
from src.train import train_gpr_model_reactive_then_proactive_per_device_type


def main():
    results = [
        'test-reactive-hetero/infra-1/results-hetero-reactive/20250423-135958-631147/0/0/peak-config.json',
        'test-reactive-hetero/infra-1/results-hetero-reactive/20250423-135958-631147/0/1/peak-config.json',
        'test-reactive-hetero/infra-1/results-hetero-reactive/20250423-135958-631147/0/2/peak-config.json',
        'test-reactive-hetero/infra-1/results-hetero-reactive/20250423-135958-631147/0/3/peak-config.json',

    ]
    # with open(results_file, 'r') as fd:
    #     results = json.load(fd)
    # print(len(results))
    encoder = get_platform_type_encoder()

    train_gpr_model_reactive_then_proactive_per_device_type(results, False, encoder=encoder)


if __name__ == '__main__':
    main()
