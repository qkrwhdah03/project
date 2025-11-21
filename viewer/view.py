import argparse

class Parser:
    def __init__(self,)-> None:
        self.parser = argparse.ArgumentParser()


    def parser(self,)-> argparse.Namespace:
        args = self.parser.parse_args()
        return args

if __name__ == "__main__":

    args = Parser().parse()
