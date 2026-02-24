from common.config import Config
from datasets.edbt.data_generator import EdbtDataGenerator

def main():
    config = Config.load()
    generator = EdbtDataGenerator(config)

    generator.run()

if __name__ == '__main__':
    main()
