# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

HINT_SEQUENCE_QUERY = """Hint: format your query as, e.g., 
<pre>gene:GENEID;split:0,1000</pre> or
<pre>ensembl:ENSGXXXX;split:0,1000</pre>.

This will trim the sequence result from 0th to 1000th position (from 5' to 3'),
or <pre>;split:-1000,-1</pre>,
to do the same from 3' to 5'"""