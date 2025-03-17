from epyt import epanet
import time

# Load a network.
d = epanet('networks/example.inp')

# Set simulation time duration.
hrs = 50
d.setTimeSimulationDuration(hrs * 3600)

# Hydraulic analysis using epanet2.exe binary file.
start_1 = time.time()
hyd_res_1 = d.getComputedTimeSeries_ENepanet()
stop_1 = time.time()
hyd_res_1.disp()

# Hydraulic analysis using epanet2.exe binary file.
start_2 = time.time()
hyd_res_2 = d.getComputedTimeSeries()
stop_2 = time.time()
hyd_res_2.disp()

# Hydraulic analysis using the functions ENopenH, ENinit, ENrunH, ENgetnodevalue/&ENgetlinkvalue, ENnextH, ENcloseH.
# (This function contains events)
start_3 = time.time()
hyd_res_3 = d.getComputedHydraulicTimeSeries()
stop_3 = time.time()
hyd_res_3.disp()

# Hydraulic analysis step-by-step using the functions ENopenH, ENinit, ENrunH, ENgetnodevalue/&ENgetlinkvalue,
# ENnextH, ENcloseH. (This function contains events)
etstep = 3600
d.setTimeReportingStep(etstep)
d.setTimeHydraulicStep(etstep)
d.setTimeQualityStep(etstep)
start_4 = time.time()
d.openHydraulicAnalysis()
d.initializeHydraulicAnalysis()
tstep, P, T_H, D, H, F = 1, [], [], [], [], []
while tstep > 0:
    t = d.runHydraulicAnalysis()
    P.append(d.getNodePressure())
    D.append(d.getNodeActualDemand())
    H.append(d.getNodeHydraulicHead())
    F.append(d.getLinkFlows())
    T_H.append(t)
    tstep = d.nextHydraulicAnalysisStep()
d.closeHydraulicAnalysis()
stop_4 = time.time()

print(f'Pressure: {P}')
print(f'Demand: {D}')
print(f'Hydraulic Head {H}')
print(f'Flow {F}')