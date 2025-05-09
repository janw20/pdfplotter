from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd


def element_to_str(
    A: int | float | None = None, Z: int | float | None = None, long: bool = False
) -> str:
    """Convert atomic number and mass number to element symbol.

    Parameters
    ----------
    A : int | float | None, optional
        Mass number of the element. By default None.
    Z : int | float | None, optional
        Atomic number of the element. By default None.
    long : bool, optional
        If the name of the element should be returned instead of its name (e.g. "Lead" instead of "Pb"). By default False.

    Returns
    -------
    str
        Symbol or name of the element.
    """
    if A is not None:
        if Z is not None:
            raise ValueError("Please pass either A or Z, not both")

        row = _elements.iloc[(_elements["AtomicMass"] - A).abs().argsort().iloc[0]]
    elif Z is not None:
        row = _elements.iloc[
            pd.Series(_elements.index.get_level_values("AtomicNumber") - Z)
            .abs()
            .argsort()
            .iloc[0]
        ]
    else:
        raise ValueError("Please pass either A or Z")

    if long:
        return row["Name"]
    else:
        return row["Symbol"]


_elements_csv = """"AtomicNumber","Symbol","Name","AtomicMass"
1,"H","Hydrogen",1.0080
2,"He","Helium",4.00260
3,"Li","Lithium",7.0
4,"Be","Beryllium",9.012183
5,"B","Boron",10.81
6,"C","Carbon",12.011
7,"N","Nitrogen",14.007
8,"O","Oxygen",15.999
9,"F","Fluorine",18.99840316
10,"Ne","Neon",20.180
11,"Na","Sodium",22.9897693
12,"Mg","Magnesium",24.305
13,"Al","Aluminum",26.981538
14,"Si","Silicon",28.085
15,"P","Phosphorus",30.97376200
16,"S","Sulfur",32.07
17,"Cl","Chlorine",35.45
18,"Ar","Argon",39.9
19,"K","Potassium",39.0983
20,"Ca","Calcium",40.08
21,"Sc","Scandium",44.95591
22,"Ti","Titanium",47.867
23,"V","Vanadium",50.9415
24,"Cr","Chromium",51.996
25,"Mn","Manganese",54.93804
26,"Fe","Iron",55.84
27,"Co","Cobalt",58.93319
28,"Ni","Nickel",58.693
29,"Cu","Copper",63.55
30,"Zn","Zinc",65.4
31,"Ga","Gallium",69.723
32,"Ge","Germanium",72.63
33,"As","Arsenic",74.92159
34,"Se","Selenium",78.97
35,"Br","Bromine",79.90
36,"Kr","Krypton",83.80
37,"Rb","Rubidium",85.468
38,"Sr","Strontium",87.62
39,"Y","Yttrium",88.90584
40,"Zr","Zirconium",91.22
41,"Nb","Niobium",92.90637
42,"Mo","Molybdenum",95.95
43,"Tc","Technetium",96.90636
44,"Ru","Ruthenium",101.1
45,"Rh","Rhodium",102.9055
46,"Pd","Palladium",106.42
47,"Ag","Silver",107.868
48,"Cd","Cadmium",112.41
49,"In","Indium",114.818
50,"Sn","Tin",118.71
51,"Sb","Antimony",121.760
52,"Te","Tellurium",127.6
53,"I","Iodine",126.9045
54,"Xe","Xenon",131.29
55,"Cs","Cesium",132.9054520
56,"Ba","Barium",137.33
57,"La","Lanthanum",138.9055
58,"Ce","Cerium",140.116
59,"Pr","Praseodymium",140.90766
60,"Nd","Neodymium",144.24
61,"Pm","Promethium",144.91276
62,"Sm","Samarium",150.4
63,"Eu","Europium",151.964
64,"Gd","Gadolinium",157.2
65,"Tb","Terbium",158.92535
66,"Dy","Dysprosium",162.500
67,"Ho","Holmium",164.93033
68,"Er","Erbium",167.26
69,"Tm","Thulium",168.93422
70,"Yb","Ytterbium",173.05
71,"Lu","Lutetium",174.9668
72,"Hf","Hafnium",178.49
73,"Ta","Tantalum",180.9479
74,"W","Tungsten",183.84
75,"Re","Rhenium",186.207
76,"Os","Osmium",190.2
77,"Ir","Iridium",192.22
78,"Pt","Platinum",195.08
79,"Au","Gold",196.96657
80,"Hg","Mercury",200.59
81,"Tl","Thallium",204.383
82,"Pb","Lead",207.97665
83,"Bi","Bismuth",208.98040
84,"Po","Polonium",208.98243
85,"At","Astatine",209.98715
86,"Rn","Radon",222.01758
87,"Fr","Francium",223.01973
88,"Ra","Radium",226.02541
89,"Ac","Actinium",227.02775
90,"Th","Thorium",232.038
91,"Pa","Protactinium",231.03588
92,"U","Uranium",238.0289
93,"Np","Neptunium",237.048172
94,"Pu","Plutonium",244.06420
95,"Am","Americium",243.061380
96,"Cm","Curium",247.07035
97,"Bk","Berkelium",247.07031
98,"Cf","Californium",251.07959
99,"Es","Einsteinium",252.0830
100,"Fm","Fermium",257.09511
101,"Md","Mendelevium",258.09843
102,"No","Nobelium",259.10100
103,"Lr","Lawrencium",266.120
104,"Rf","Rutherfordium",267.122
105,"Db","Dubnium",268.126
106,"Sg","Seaborgium",269.128
107,"Bh","Bohrium",270.133
108,"Hs","Hassium",269.1336
109,"Mt","Meitnerium",277.154
110,"Ds","Darmstadtium",282.166
111,"Rg","Roentgenium",282.169
112,"Cn","Copernicium",286.179
113,"Nh","Nihonium",286.182
114,"Fl","Flerovium",290.192
115,"Mc","Moscovium",290.196
116,"Lv","Livermorium",293.205
117,"Ts","Tennessine",294.211
118,"Og","Oganesson",295.216 """
_elements = pd.read_csv(StringIO(_elements_csv), index_col="AtomicNumber")
