# L4 source expansion audit — 8 October 2026

## Finding

The current IPCC AR5 Annex III Table A.III.2 pack contains five technology
records from one report/table, so it is one source-world. The report points to
Annex II A.II.9.3 for underlying literature. More question wordings, rows,
technologies or scenarios from this same summary do not create independent
source-worlds. See the [official IPCC Annex III](https://archive.ipcc.ch/pdf/assessment-report/ar5/wg3/ipcc_wg3_ar5_annex-iii.pdf)
and [Annex II](https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-ii.pdf).

An independent source audit found several bounded retrieval candidates, but no
single comparable collection with 20 independent study groups:

- [NREL geothermal systematic review](https://docs.nrel.gov/docs/fy17osti/68474.pdf),
  Table A-6 summarizes estimates from 29 screened studies. Some studies have
  multiple rows/cases with the same author/year; report rows must be grouped by the underlying study,
  and shared-inventory dependence remains. Do not count 29 rows as 29 worlds.
- [NREL 2024 utility-scale PV LCA](https://www.nrel.gov/docs/fy24osti/87372.pdf),
  Table B6, describes six supply-chain/EOL cases in one study. Its caption says
  kgCO2e/kWh while the table header and Figure 10 text say gCO2e/kWh; preserve
  the discrepancy as unresolved. These are scenarios, not confidence intervals.
- [NETL U.S. 2020 natural-gas profile](https://www.netl.doe.gov/projects/files/LifeCycleAnalysisofNaturalGasExtractionandPowerGenerationUS2020EmissionsProfile_012425.pdf),
  Exhibits 7-1 to 7-3, reports multiple plant applications within one study.
  Page 118 says peaking and baseload cases are not directly comparable. Its
  upstream error bars do not describe full-total uncertainty.
- [EDF French nuclear-fleet LCA](https://www.edf.fr/sites/groupe/files/2022-06/edfgroup_acv-4_etude_20220616.pdf),
  Table 14, reports a bounded national fleet result from one study. Reactors
  and sites are not independent publications; its uncertainty is not a paired
  95% interval.
- [JRC 2017 country electricity factors](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC107518/jrc_technical_reports_-_com_default_emission_factors-2017.pdf),
  Annex I.4 Table AI.4.3, shares IEA energy balances and lifecycle factors
  across countries/years. Its consumption allocation differs from IPCC
  technology-specific lifecycle medians.
- [JRC 2024 NEEFE dataset](https://data.jrc.ec.europa.eu/dataset/919df040-0252-4e4e-ad82-c054896e1641)
  and [methodology report](https://publications.jrc.ec.europa.eu/repository/handle/JRC136272)
  distinguish activity-only CO2, activity CO2/CH4/N2O, and lifecycle GHG fields.
  Country/year rows and successive releases do not establish independent
  studies. JRC notes the three measures must not be mixed.

## IPCC underlying-study trace

[IPCC AR5 Annex III, Table A.III.2](https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf#page=7)
lists 17 technology entries. Footnote i sends readers to [Annex II,
A.II.9.3](https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-ii.pdf#page=26)
for the underlying literature and methods. The bibliography names studies for
each category, for example:

| Category | References named in Annex II | Qualification issue |
| --- | --- | --- |
| Coal/gas, including CCS | Burnham (2012), Singh (2011a/b), Corsten (2013), NETL (2010a–e), Hertwich (2013) | Totals combine common component inventories and Monte Carlo assumptions. |
| Nuclear | Lenzen (2008); Warner & Heath (2012) | Only a few cited studies for this technology group. |
| CSP / PV / wind | Viebahn, Burkhardt, Whitaker, Hsu, Kim, Arvesen & Hertwich, Dolan & Heath | Studies may share reviews, background inventories, or harmonization inputs. |
| Ocean / hydro | Parker, Walker & Howell, Douglas, Woollcombe-Adams, Kelly, Kumar, Raadal, Moreau, Kemenes, Barros | Table rows do not map one-to-one to a single source estimate. |
| Biomass/geothermal | Multiple references and methodology discussions | Annex II does not fully attribute every table statistic to independent source records. |

These citations identify literature families; they do not establish a complete
row-to-observation crosswalk or 20 independent, like-for-like source groups.
Annex II also describes shared harmonization/background data and separately
assessed inputs. Thus, the existence of more than 20 bibliography entries is
not itself enough to qualify L4.

The next evidence-backed step is to recover the original NREL/OpenEI study-level
records cited around Annex II p.1306, link each eligible observation to its
primary study and shared inventory family, then count only distinct families
whose boundary, functional unit, lifecycle statistic and uncertainty label
match. Keep the current AR5 table grouped as one report-world until that trace
meets the source-world floor.

## Safe expansion rule

An added L4 module must retain its own source, population, year, boundary,
functional unit, statistic and uncertainty label. Group by the originating
study or shared model input, not by table row, technology, country, site,
scenario, year or wording. Compare values only inside a checked domain where
their fields and boundaries match. Keep unverified units or ambiguous source
transcriptions unavailable until resolved. Do not combine the candidate
families into one energy ranking or claim 20 worlds by duplicating table rows.

The source candidates can improve scoped retrieval coverage with clear labels,
but the independent-world floor for the current L4 claim remains unmet. No
source data was imported; no held-out suite was generated/evaluated and no
model/GPU work was run for this audit.
