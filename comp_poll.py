import pandas as pd

df = pd.read_excel(r'\\wsl.localhost\Ubuntu-20.04\home\miriam27\PAE\MAPPO\1.A.3.b.i-iv Road Transport Appendix 4 Emission Factors 2021.xlsx',hoja = 'HOT_EMISSIONS_PARAMETERS')

#SO2
#ESO2m = 2 * Ksm * FCm

df.head()
