from .wso_test import wso_test
from .epanet_test import epanet_test, epanet_wso_test

if __name__ == "__main__":
    # wso_test()
    # epanet_test("epanet/files/example.inp")
    epanet_wso_test("epanet/files/example.inp")