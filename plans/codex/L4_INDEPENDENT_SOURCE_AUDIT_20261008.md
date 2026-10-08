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

## Land-based wind candidate follow-up

[Dolan & Heath (2012)](https://onlinelibrary.wiley.com/doi/10.1111/j.1530-9290.2012.00464.x)
report 49 references and 126 estimates across onshore and offshore wind; the
[NREL harmonization fact sheet](https://www.nrel.gov/docs/fy13osti/57131.pdf)
breaks this out as 44 references and 107 estimates for land-based wind. The
paper says the reported values are harmonized to consistent gross system
boundaries and selected performance parameters, so this is a promising
single-technology source pool to investigate. The 44 references are candidate
groups only: estimates can repeat within a reference, and references can
share inventories or model inputs.

The current [NLR catalog submission 171](https://data.nlr.gov/submissions/171)
does not expose those 107 source estimates: its 28 KB workbook is described as
quartile estimates by lifecycle stage and technology. The paper lists
Supporting Information S1 as further detail on screening and harmonization,
but its exact `JIEC_464_sm_suppmat.pdf` (222.5 KB) download returned HTTP 403.
The modern and legacy OpenEI LCA app routes returned HTTP 503; Wayback queries
returned HTTP 429, which does not establish that the files were never
archived. The current [OEDI submission 8228](https://data.openei.org/submissions/8228)
is the same NREL quartile-summary dataset, not the old study-level export.
Thus far I did not find an accessible, machine-readable
estimate-to-reference-to-inventory crosswalk. Therefore the 44-reference
count cannot yet be used as 44 independent worlds, and no wind data was
imported into L4.

An [accessible NY DPS-hosted copy of the article](https://documents.dps.ny.gov/public/Common/ViewDoc.aspx?DocRefId=%7B309B2270-5EF1-4DE1-9B56-A4A1D1A6F4C9%7D&DocTitle=Avian_25_Dolan+Heath_2012_LC+GHG+emissions+of+utility-scale+wind+power)
exposes Table 1's author/year, onshore/offshore classification, and study
characteristics. The paper states that Supporting Information Table S3 contains
individual estimates by publication for each harmonization step. This narrows
the recovery target to S1 Table S3 joined to Table 1 and the primary papers;
the accessible copy does not include S1 itself.

Table 1 and the bibliography yield 44 pure onshore author/year references.
The following accounting is only a bibliographic audit queue:

| References | Proposed handling | Count effect |
| --- | --- | ---: |
| White & Kulcinski 1998; White & Kulcinski 2000; White 2006 | Explicit updates and repeated Buffalo Ridge/Glenmore systems; one provisional family | 3→1 |
| Dones et al. 2005; Dones et al. 2007; Jungbluth et al. 2005 | Repeated European 0.8-MW system and Ecoinvent v2.0; provisional inventory family | 3→1 |
| Pacca & Horvath 2002; Pacca 2003 | Shared lead author, repeated 0.6-MW system, thesis variants; provisional family | 2→1 |
| Vestas Wind Systems 2006a; 2006b | Same manufacturer and related declarations, different turbine models; provisional family | 2→1 |
| Berry et al. 1998; European Commission 1995 | Overlapping Welsh sites, different capacity assumptions; provisional family | 2→1 |
| Jacobson 2009; WEC 2004; van de Vate 1996 | Set aside from a primary-study candidate count pending inspection; only Jacobson is explicitly titled a review | 37→34 |

The first, explicitly documented update family reduces 44 to 42; applying all
four additional provisional merges gives 37. Setting aside the three
synthesis/compilation-risk references yields 34 candidate families. The 32
remaining singleton references are Ardente et al. 2008; Chataignere & Le
Boulch 2003; Crawford 2009; Enel 2004; Frischknecht 1998; Hartmann 1997;
Hondo 2005; Jacobson 2009; Khan 2005; Krewitt 1997; Kuemmel & Sørensen 1997;
Lee & Tzeng 2008; Lenzen & Wachsmann 2004; Liberman & LaPuma 2003; Martínez
et al. 2009; McCulloch et al. 2000; Nadal 1995; Pehnt 2006; Proops et al. 1996;
Rule et al. 2009; Rydh et al. 2004; Schleisner 2000; SECDA 1994; Spitzley &
Keoleian 2004; Tremeac & Meunier 2009; Uchiyama 1996; van de Vate 1996;
Vattenfall 2003; Voorspools et al. 2000; Waters et al. 1997; WEC 2004; and
Wibberly 2001. This count makes 20 plausible as a candidate-audit target, not
as 20 independent or qualified worlds.

The paper's p. 16 “Clustering Bias” discussion says author/study clustering and
common Ecoinvent data were not quantified, while the analysis treats all 126
estimates independently. It also notes shared median additions for missing
lifecycle stages (pp. 10–11). Its Table 2 gives aggregate statistics; Figure 3
shows ordinal trajectories without per-study numeric labels. The article says
S1 Table S3 contains the per-publication values for each harmonization step.
Thus the body does not provide reference-linked values for the 107 land-based
estimates. Recovering S1 would enable that join, but the candidate families
must still be deduplicated against shared inventories before they count toward
L4's 20-group floor.

The [DOE Wind Vision 2015 Appendix J](https://www.energy.gov/documents/windvisionstudyfinalappendixbook-web150pdf)
adds a partial bibliography: Table J-2 lists study characteristics for later
wind-LCA updates to Dolan and Heath, while explicitly referring readers to the
2012 review for earlier studies. It does not supply the 2012 Table S3 values
or complete the estimate-to-inventory crosswalk, and is not an independent
replacement for that review.

Smallest next step: recover the old OpenEI land-based wind export or the paper's
study-level supplemental table, then join each estimate to its primary study
and shared background inventory family. Count only the surviving groups after
checking that the L4 boundary, functional unit, statistic, and uncertainty
labels match. Until then, keep the harmonization report as one source-world.

A newer [IEA 2024 wind review](https://iea.blob.core.windows.net/assets/3ff59090-1420-4e74-9069-a90602e56bf3/IEAupstreamlifecycleemissionfactors_2024.pdf#page=22)
starts from roughly 120 onshore-wind studies and retains about 50 estimates.
It harmonizes GWP, included gases, capacity factor, and lifetime, but the
report exposes only aggregate min/quartile/median/max results in [Annex III](https://iea.blob.core.windows.net/assets/3ff59090-1420-4e74-9069-a90602e56bf3/IEAupstreamlifecycleemissionfactors_2024.pdf#page=49).
Its references section identifies source frameworks and data products rather
than mapping each retained estimate to a primary LCA and shared inventory.
Thus the roughly 50 values remain one IEA review-world for L4, not 50
independent groups; the row-level crosswalk is still the next required input.

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
