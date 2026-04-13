import os

def clean_tree_operators(content):
    # Remove Tree Operators to fix topology for parity
    content = content.replace("<operator id='treeScaler' spec='ScaleOperator' scaleFactor=\"0.5\" weight=\"1\">\n            <tree idref=\"tree\"/>\n        </operator>", "")
    content = content.replace("<operator spec='Uniform' weight=\"10\">\n            <tree idref=\"tree\"/>\n        </operator>", "")
    content = content.replace("<operator spec='SubtreeSlide' weight=\"5\" gaussian=\"true\" size=\"1.0\">\n            <tree idref=\"tree\"/>\n        </operator>", "")
    content = content.replace("<operator id='narrow' spec='Exchange' isNarrow='true' weight=\"1\">\n            <tree idref=\"tree\"/>\n        </operator>", "")
    content = content.replace("<operator id='wide' spec='Exchange' isNarrow='false' weight=\"1\">\n            <tree idref=\"tree\"/>\n        </operator>", "")
    content = content.replace("<operator spec='WilsonBalding' weight=\"1\">\n            <tree idref=\"tree\"/>\n        </operator>", "")
    return content

def create_model1_xml():
    with open("primate.xml", "r") as f:
        content = f.read()

    # Set shorter chains
    content = content.replace('chainLength="1000000"', 'chainLength="500000"')

    # Replace clock model (Strict Clock)
    content = content.replace(
        """    <input spec='UCRelaxedClockModel' id="branchRates" normalize='true'>
        <!--<parameter name='mutationRate'>1.0</parameter>-->
		<parameter name='clock.rate' id='ucld.mean' value='1.0'/>
        <distr id='lognormal' spec="beast.base.inference.distribution.LogNormalDistributionModel">
            <parameter name='M' id='M' value="1"/>
            <parameter name='S' id='ucld.stdev' value="0.5" lower="0" upper="5"/>
        </distr>
        <parameter spec='IntegerParameter' name='rateCategories' id='rateCategories' dimension="11" value="1"/>
        <input name='tree' idref="tree"/>
    </input>""",
        """    <input spec='StrictClockModel' id="branchRates">
        <parameter name='clock.rate' id='clockRate' value='1.0'/>
    </input>"""
    )
    
    content = content.replace("<input name='stateNode' idref='ucld.mean'/>\n", "")
    content = content.replace("<input name='stateNode' idref='ucld.stdev'/>\n", "")
    content = content.replace("<input name='stateNode' idref='rateCategories'/>\n", "")
    
    # State Node (add clockRate to state)
    content = content.replace(
        "            <input name='stateNode' idref='tree'/>",
        "            <input name='stateNode' idref='clockRate'/>\n            <input name='stateNode' idref='tree'/>"
    )

    # Remove Operators for UCLN
    content = content.replace(
        """        <operator id="categoriesRandomWalk" spec="UniformOperator" weight="1">
            <input name="parameter" idref="rateCategories"/>
        </operator>""",
        ""
    )
    content = content.replace(
        "        <operator id='SScaler' spec='ScaleOperator' scaleFactor=\"0.5\" weight=\"1\" parameter='@ucld.stdev'/>\n",
        ""
    )
    
    content = clean_tree_operators(content)

    # Filenames
    content = content.replace('fileName="test.$(seed).log"', 'fileName="beast_model1_gamma.log"')
    content = content.replace('fileName="test.(time).$(seed).trees"', 'fileName="beast_model1_gamma.trees"')
    content = content.replace('fileName="test.(subst).$(seed).trees"', 'fileName="beast_model1_gamma_subst.trees"')

    with open("Model_1_Spatial_Heterogeneity/beast_model1_gamma.xml", "w") as f:
        f.write(content)
        
def create_model2_xml():
    with open("primate.xml", "r") as f:
        content = f.read()

    # Set shorter chains
    content = content.replace('chainLength="1000000"', 'chainLength="500000"')

    # Fix filenames
    content = content.replace('fileName="test.$(seed).log"', 'fileName="beast_model2_ucln.log"')
    content = content.replace('fileName="test.(time).$(seed).trees"', 'fileName="beast_model2_ucln.trees"')
    content = content.replace('fileName="test.(subst).$(seed).trees"', 'fileName="beast_model2_ucln_subst.trees"')

    # Fix tree
    content = clean_tree_operators(content)

    with open("Model_2_Temporal_Heterogeneity/beast_model2_ucln.xml", "w") as f:
        f.write(content)

if __name__ == "__main__":
    create_model1_xml()
    create_model2_xml()
    print("XMLs created.")
