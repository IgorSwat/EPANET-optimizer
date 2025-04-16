from .wso_test import wso_test, wso_vs_pso_test
from .epanet_test import epanet_test, epanet_wso_test

if __name__ == "__main__":
    # wso_test()
    wso_vs_pso_test()
    # epanet_test("epanet/files/example.inp")
    # epanet_wso_test("epanet/files/example.inp")