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
The following accounting is a candidate audit queue, not a verified
independence partition:

| References | Evidence and handling | Count effect |
| --- | --- | ---: |
| White & Kulcinski 1998; White & Kulcinski 2000; White 2006 | Explicit update lineage and repeated Buffalo Ridge/Glenmore systems; one candidate family. [White 2006 publisher record](https://link.springer.com/article/10.1007/s11053-007-9024-y) | 3→1 |
| Dones et al. 2005; Dones et al. 2007; Jungbluth et al. 2005 | Jungbluth §2.2–2.3, pp. 8–9, and Dones 2007 §§13.2–13.6, pp. 143–152, document the same ecoinvent/Burger & Bauer foreground. Dones 2005 Table 1, p. 7, is ecoinvent-based with a matching 800-kW case, so joining all three is conservative. [Jungbluth](https://esu-services.ch/fileadmin/download/jungbluth-2004-PV-wind.pdf); [Dones 2007](https://ecolo.org/documents/documents_in_english/Life-cycle-analysis-PSI-05.pdf); [Dones 2005](https://bravenewclimate.com/files/expolwp6.pdf) | 3→1 |
| Pacca & Horvath 2002; Pacca 2003; Schleisner 2000 | Pacca thesis p. 61 reproduces Chapter 4 from the article; §4.3.1, p. 106, sources its Utah turbine from Schleisner. Merge as a shared-foreground-input block, not identical complete LCIs. [Pacca & Horvath](https://doi.org/10.1021/es0155884); [Pacca thesis](https://rael.berkeley.edu/old_drupal/sites/default/files/old-site-files/diss/Sergio_Pacca_2003.pdf) | 3→1 |
| Vestas Wind Systems 2006a; 2006b | V82 report pp. 6, 77 explicitly uses earlier V80/V90 LCAs as a basis; shared-model lineage, though not proven identical inventories. Only lineage was inspected; no Vestas measurements were imported. [Vestas V82 report](https://www.vestas.com/content/dam/vestas-com/global/en/sustainability/reports-and-ratings/lcas/LCA%20V82165%20MW%20onshore2007.pdf.coredownload.inline.pdf) | 2→1 |
| Berry et al. 1998; European Commission 1995 | Overlapping Welsh sites, different capacity assumptions; provisional family | 2→1 |

Starting at 44, these five blocks reduce the candidate count to 36. Jacobson
2009, WEC 2004, and van de Vate 1996 are then set aside as synthesis or
compilation risks, leaving **33 candidate families**. Only Jacobson is
explicitly titled a review; WEC and van de Vate need inspection before
excluding them from a primary-source count. The 28 other singleton references
are Ardente et al. 2008; Chataignere & Le Boulch 2003; Crawford 2009; Enel
2004; Frischknecht 1998; Hartmann 1997; Hondo 2005; Khan 2005; Krewitt 1997;
Kuemmel & Sørensen 1997; Lee & Tzeng 2008; Lenzen & Wachsmann 2004; Liberman &
LaPuma 2003; Martínez et al. 2009; McCulloch et al. 2000; Nadal 1995; Pehnt
2006; Proops et al. 1996; Rule et al. 2009; Rydh et al. 2004; SECDA 1994;
Spitzley & Keoleian 2004; Tremeac & Meunier 2009; Uchiyama 1996; Vattenfall
2003; Voorspools et al. 2000; Waters et al. 1997; and Wibberly 2001. This
count makes 20 plausible as a candidate-audit target, not 20 independent or
qualified worlds; broader ExternE/Ecoinvent overlaps may reduce it further.

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
wind-LCA updates to Dolan and Heath (citation IDs [4]–[10]), while explicitly
referring readers to the 2012 review for earlier studies. It does not supply
the 2012 Table S3 values or complete the estimate-to-inventory crosswalk, and
is not an independent replacement for that review.

## Primary-source feasibility screen

Five candidate singleton sources were checked for whether their original
reporting can enter a common, complete lifecycle GHG/kWh cohort without
inventing omitted phases or gases:

| Study | Boundary/data finding | L4 disposition |
| --- | --- | --- |
| McCulloch et al. 2000, *Life-Cycle Value Assessment of a Wind Turbine* ([publisher PDF](https://www.pembina.org/reports/windlcva.pdf)) | Reports CO2-equivalent per 1,000-kWh functional unit, but omits manufacturing-process energy and decommissioning. | No-go for a complete upstream/ongoing/downstream boundary without imputation. |
| Crawford 2009, DOI 10.1016/j.rser.2009.07.008 ([paper mirror](https://www.windland.ch/doku_wind/klima/Life%20Cycle%20Emissions%20from%20Wind%20-%20Crawford%202009.pdf)) | Tables 7–8 report embodied emissions per turbine; decommissioning/recycling are excluded. GHG is estimated with a single energy-to-emissions factor, not a gas-resolved GWP inventory. | No-go; reported primary-energy equivalents cannot be used as electric kWh. |
| Rydh et al. 2004, report NEI-SE-544 ([PNNL-hosted report](https://tethys.pnnl.gov/sites/default/files/publications/Rydhetal2004.pdf)) | Reports numerical gCO2/kWh_electric and covers production through disposal, but reports CO2, not a complete CO2-equivalent inventory including CH4/N2O. | No-go for full GHG; scope is useful only as a separate CO2 fact. |
| Martínez et al. 2009, DOI 10.1007/s11367-008-0033-9 ([paper](https://www.researchgate.net/publication/240139630_Life-cycle_assessment_of_a_2MW_rated_power_wind_turbine_CML_method)) | Reports GWP100 per kWh with manufacturing, transport, maintenance and end-of-life, but excludes grid connection/substation. Explicit Ecoinvent 1.3 background. | Conditional for its own historical scope; a shared Ecoinvent family and the omitted grid assets prevent counting it as an independent, same-boundary world. |
| Spitzley & Keoleian 2004, CSS04-05R ([institutional record](https://css.umich.edu/publications/research-publications/life-cycle-environmental-and-economic-assessment-willow-biomass)) | The institution says the report is unavailable pending correction; its linked article is about willow biomass, not the wind report. | Access no-go; do not substitute the linked article. |

This five-source screen yields **zero unconditionally admissible common-scope
groups and one conditional historical-scope candidate**. It does not show that
20 groups are impossible in the remaining references; their source boundaries,
values and background-inventory overlap still need review. No values from this
screen were transcribed or imported.

### Second primary-source batch

Five additional singleton references were screened against the same complete
common-scope requirement; a reported lifecycle result alone does not establish
complete phase/gas coverage or independent inventory provenance.

| Study | Original-source evidence and dependencies | L4 disposition |
| --- | --- | --- |
| Ardente et al. 2008, DOI 10.1016/j.rser.2006.05.013 ([author-uploaded paper](https://www.researchgate.net/publication/222226993_Energy_performances_and_life_cycle_assessment_of_an_Italian_wind_farm)) | Table 5, p. 212, reports GWP per generated electric kWh. Sections 2.2–2.6, pp. 206–209, cover production through an end-of-life scenario, but p. 209 leaves roads, buildings and foundations in place. Manufacturing and disposal assumptions explicitly use ECLIPSE/Chataignere & Le Boulch; references 18–19, p. 217, also identify Jungbluth/Frischknecht background data. The complete gas vector and exact GWP vintage were not recovered. | Conditional for the original historical scenario only; no complete common-scope admission or independent inventory established. |
| Frischknecht 1998, ETH dissertation 12599, ISBN 3-9520661-3-3 ([author-hosted thesis](https://esu-services.ch/fileadmin/download/frischknecht-1998-PHD.pdf)) | Section 9.3.10, pp. 148–149, and Table 9.11, p. 149, report selected lifecycle CO2/CH4 mass flows per TJ of electricity, not a complete gas inventory or phase-resolved GWP result. The table directs full documentation to Frischknecht et al. 1996a, Part XIII, *Windkraft*, p. 6 onward. | No-go from the thesis alone; missing gases/stages cannot be set to zero, and earlier inventory reuse is explicit. |
| Hartmann 1997, *FENCH-analysis of electricity generation greenhouse gas emissions from solar and wind power in Germany* | The exact original paper was not recovered. The official [IPCC SRREN bibliography](https://www.ipcc-wg3.de/report/IPCC_SRREN_Annex_II.pdf) identifies the IAEA meeting in Vienna, 21–24 October 1996, pp. 77–87; no DOI was identified. | Access no-go; phase/gas coverage and shared inputs remain unverified. |
| Hondo 2005, DOI 10.1016/j.energy.2004.07.020 ([original-paper text mirror](https://www.scribd.com/document/811846570/t7-Gas-Emmision-2005-SD-Hiroki-Hondo)) | Section 2.1, p. 2044, focuses on CO2/CH4; methane leakage is limited to fuels directly burned in generation. Table 9 is on **p. 2053, not p. 2052**, and reports wind CO2 per net electric kWh, separated into construction and operation. A wind-specific downstream contribution was not recovered. Earlier Hondo et al. reports [9–10] supply method/data; scenario variants share that model. | No-go for complete multi-gas lifecycle GHG; do not promote the selected-gas scope to full GHG. |
| Lenzen & Wachsmann 2004, DOI 10.1016/S0306-2619(03)00105-3 ([publisher sections](https://www.sciencedirect.com/science/article/abs/pii/S0306261903001053); [author PDF](https://isa.org.usyd.edu.au/education/documents/ISA_Wind_turbine_LCA_2004.pdf)) | The original p. 119 abstract/introduction defines an energy/CO2 analysis; Table 1, p. 125, supplies annual electric output. The publisher's “Data sources” section explicitly adopts German E-40 technical/site/cost inputs from Pick & Wagner [23–24] and translates them into Brazil. Full PDF retrieval failed; complete phase/gas coverage was not verified. | No-go for full GHG on recovered evidence; country/manufacturing/site variants remain one shared foreground/model family. |

This second batch yields **zero candidates meeting complete common-scope
admission**. Ardente supports conditional retrieval of its own reported scenario,
but cannot yet enter a re-harmonized independent cohort. The evidence does not
prove that 20 groups are impossible elsewhere; unresolved gas/GWP coverage,
end-of-life boundaries, primary-source access and inventory crosswalks remain
the blockers. No numerical values were transcribed or imported, no repository
data was changed, and no held-out, model, GPU or paid resource was used.

### Third primary-source batch

The next five references were screened for original-source access, electric-kWh
normalization, gas/GWP interpretation, phase boundary and inventory lineage.
Unknown fields below remain unknown; review summaries do not replace the
unrecovered originals.

| Study | Availability, scope and dependency evidence | L4 disposition |
| --- | --- | --- |
| Chataignere & Le Boulch 2003, *Wind Turbine (WT) Systems*, EDF R&D/ECLIPSE, 110 pp. | The exact report was not recovered; its former [project download page](https://www.eclipse-eu.org/pubres_guide.html) now serves unrelated content. The [official IPCC bibliography](https://www.ipcc-wg3.de/report/IPCC_SRREN_Annex_II.pdf) establishes the report identity, not its functional unit, complete gases/GWP or phase coverage. Ardente 2008, pp. 206, 209, and Tremeac & Meunier's manuscript section 2.1 explicitly reuse its manufacturing, operation or disposal inputs. | Access no-go. It is a dependency root to trace, not a new verified independent world. |
| Tremeac & Meunier 2009, DOI 10.1016/j.rser.2009.01.001 ([author-uploaded manuscript](https://www.researchgate.net/publication/240423604_Life_cycle_analysis_of_45_MW_and_250_W_wind_turbines)) | Manuscript sections 1.1–2.1, pp. 4–8, define one produced electric kWh, construction/operation/decommissioning, and Impact 2002+ with IPCC 2001 **500-year GWP**, not GWP100. Tables 2/4 report climate impact/intensity, but the complete gas vector was not recovered. The 4.5-MW inputs explicitly reuse ECLIPSE and Vestas; operation and decommissioning also use assumed energy relationships. | Conditional retrieval of the original modeled scope only. Preserve its CO2-labelled intensity versus CO2-equivalent climate category; do not relabel the aggregate as GWP100 or count turbine/transport scenarios as independent sources. |
| Pehnt 2006, DOI 10.1016/j.renene.2005.03.002 ([original-paper text mirror](https://www.scribd.com/document/175926226/Pehnt-Dynamic-LCA); [publisher](https://www.sciencedirect.com/science/article/pii/S0960148105000662)) | Section 2.1, pp. 57–58, defines one electric kWh at the plant under modeled German 2010 conditions; production, operation/maintenance and recycling/disposal are included. Distribution and lorry-road utilization are outside the boundary; closed-loop recycling is assumed. Table 1, p. 58, declares CO2/CH4/N2O GWP100; Table 3, p. 60, has species-resolved totals. It is explicitly streamlined, with other technology-specific substances screened. IFEU/Umberto background and Hartmann's 2001 thesis [2] supply inputs. | Conditional for the declared streamlined historical scope; species-level recharacterization is a possible next check after verifying table alignment and required-gas coverage. Shared inputs and the modeled reference prevent automatic independent/common-scope admission. |
| Vattenfall 2003, *Certified Environmental Product Declaration of Electricity from Vattenfall AB's Swedish Windpower Plants* | The exact 2003 EPD was not recovered; the historical [registry 044 route](https://www.environdec.com/reg/044/) was inaccessible. The report is identified in the official IPCC bibliography, but original electric-kWh definition, gases/GWP, lifecycle inclusions and background inventory could not be verified. Later Vattenfall declarations are different versions/populations. | Access no-go. Do not substitute a later declaration or count constituent turbines/sites as verified source groups. |
| Voorspools et al. 2000, DOI 10.1016/S0306-2619(00)00016-7 ([institutional original abstract](https://www.mech.kuleuven.be/en/tme/research/energy_environment/pdf/2000P41_abstract)) | The one-page KU Leuven abstract was recovered; the full pp. 307–330 article was not. It reports electric-kWh-normalized emissions and describes process-chain/IO methods for construction, maintenance and demolition. It does not establish the gas set, GWP horizon/vintage, detailed stage boundary or inventory lineage. The two methods concern the same underlying investment systems. | Access/scope no-go pending the full article and inventory crosswalk; the abstract's GHG wording is not evidence of a complete compatible gas inventory. |

This batch adds **zero admitted independent common-scope groups**. Pehnt is a
conditional candidate for a clearly bounded, species-resolved historical module;
Tremeac & Meunier supports conditional retrieval under its distinct 500-year
interpretation. Neither establishes a 20-source cohort. Original-report access,
required-gas and boundary matching, table verification and shared-inventory
crosswalks remain the limits. No numerical emissions values were transcribed or
imported, no repository data was changed, and no held-out, GPU or paid resource
was accessed. No tests were run for this documentation-only update.

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
The associated [IEA 2024 database product](https://www.iea.org/data-and-statistics/data-product/life-cycle-upstream-emissions-factors-2024-2)
lists its XLS as account/IP access under a non-CC licence and describes
country-level factors weighted by national generation mix. That product does
not provide a freely accessible study-level crosswalk here; no IEA data file
was obtained or imported.

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

### 8 October 2026 — NLR/OEDI and publisher supplement recovery check

The current [NLR catalog record 171](https://data.nlr.gov/submissions/171),
DOI [10.7799/1819907](https://doi.org/10.7799/1819907), is marked Public. Its
data-resource history exposes one version (v1, dated 2021-08-23): a 28 KB
`EF_Table_FINAL.xlsx` described as technology-level quartile estimates by
lifecycle phase, in g CO2e/kWh. The listed phases are one-time upstream,
ongoing combustion, ongoing non-combustion, one-time downstream, and total.
The catalog page was last updated 2026-05-22, but its resource history does not
show a newer study-level file. [OEDI submission 8228](https://data.openei.org/submissions/8228)
has the same dataset title, authors, description and date, and its citation
points back to NLR record 171; it is a mirror/catalog representation of that
summary record, not an independent or row-level export.

Wiley's [Dolan and Heath article record](https://onlinelibrary.wiley.com/doi/10.1111/j.1530-9290.2012.00464.x)
lists Supporting Information S1, filename `JIEC_464_sm_suppmat.pdf` (222.5 KB),
described as containing GHG emissions by lifecycle and harmonization stage.
The article identifies electricity-normalized GHG results and reports
harmonization to gross system boundaries and selected parameters. The [NREL
wind factsheet](https://www.nrel.gov/docs/fy13osti/57131.pdf) confirms a
land-based subgroup and its reference/estimate counts. However, the listed
supplement's publisher download endpoint
([Wiley downloadSupplement](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fj.1530-9290.2012.00464.x&file=JIEC_464_sm_suppmat.pdf))
returns HTTP 403. Accordingly, S1 Table S3's rows remain unavailable for
joining to the article's author/year and study-characteristics table and then
to primary studies and shared inventory families. Gas-vector/GWP details and
dependence cannot be verified for those rows from the accessible catalog
metadata. No source-world can be admitted on this evidence; the exact blocker
is the missing estimate-to-reference-to-primary-study/inventory crosswalk,
with the publisher-hosted supplement listed but access-denied. No numeric
emissions values were transcribed or imported.

### 8 October 2026 — exact S1 repository and metadata trace

Follow-up checks found no separate legitimate deposit of the publisher's exact
S1 file. The [NLR Research Hub record](https://research-hub.nlr.gov/en/publications/life-cycle-greenhouse-gas-emissions-of-utility-scale-wind-power-s-4/)
lists the article DOI as its access-to-document link and exposes no S1
attachment. The [DOE OSTI record 1040937](https://www.osti.gov/biblio/1040937)
and its [official record API](https://www.osti.gov/api/v1/records/1040937)
identify the article/report number but expose no full text or supplement
relation. The [Crossref DOI record](https://api.crossref.org/works/10.1111%2Fj.1530-9290.2012.00464.x)
has an empty `relation` object and article-only links; the [DataCite related-DOI
query](https://api.datacite.org/dois?query=relatedIdentifiers.relatedIdentifier%3A10.1111%2Fj.1530-9290.2012.00464.x)
returns zero records. Wiley lists `JIEC_464_sm_suppmat.pdf`, but both its
[supplement download route](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fj.1530-9290.2012.00464.x&file=JIEC_464_sm_suppmat.pdf)
and [supplement-info route](https://onlinelibrary.wiley.com/doi/suppl/10.1111/j.1530-9290.2012.00464.x/supinfo)
return HTTP 403. Thus S1 Table S3 remains unavailable and cannot be joined to
primary studies or shared-inventory families; no additional source-world is
admitted. No emissions values were transcribed or imported.

### 8 October 2026 — fourth primary-source batch

Five remaining candidate references were screened for original-source access,
functional unit and boundary, directly verified gas/GWP treatment, uncertainty
statistic, and inventory-family overlap. Secondary summaries were used only to
locate primary records; no reported emissions values were transcribed or
imported.

| Study | Verified source and scope | L4 disposition |
| --- | --- | --- |
| Enel SpA 2004, S-P-00066 | The exact EPD is identified as *Certified Environmental Product Declaration of Electricity from ENEL's wind plant in Sclafani Bagni*; its historical registry PDF (`https://www.environdec.com/reg/e_epd66.pdf`) was not accessible. [Enel's official 2005 press release](https://corporate.enel.it/media/esplora-comunicati/CS/2005/12/nuovo-campo-eolico-enel-a-gangi-palermo-32-aerogeneratori-per-una-potenza-di-oltre-27-mw) confirms Sclafani Bagni's EPD was obtained in 2004, but does not expose the original boundary or impact tables. Functional unit, phases, gases/GWP, uncertainty and inventory lineage therefore remain unverified. Do not substitute Enel's current, expanded Sclafani asset population. | Access/scope no-go; no independent group admitted. |
| Khan, Hawboldt & Iqbal 2005 | The [publisher record](https://www.sciencedirect.com/science/article/pii/S0960148104001843) identifies *Life Cycle Analysis of wind–fuel cell integrated system* for remote Newfoundland and Labrador; full text is subscription restricted. It is a hybrid-system study, not a standalone wind-generator case. Functional unit, detailed boundary, gases/GWP, uncertainty and inventory lineage cannot be verified from the accessible record. Its bibliography names earlier wind work, including Voorspools and Schleisner, but citation overlap alone does not establish shared inventories. | No-go for a wind-only common cohort; no group admitted. |
| Krewitt 1997 | The citation is ambiguous. The [official Springer record](https://link.springer.com/book/10.1007/978-3-642-60914-5) identifies Friedrich and Krewitt's 1997 external-cost book, but its renewable-energy chapter is attributed to Raptis, Kaspar and Sachau. No exact Krewitt-authored primary wind-LCA record matching the audit's author-year entry was recovered. The separate European Commission [ExternE Wind & Hydro volume](https://www.externe.info/externe_d7/sites/default/files/vol6.pdf) is a 1995 report, not evidence that a 1997 Krewitt entry is an independent source. Functional unit, boundary, gas/GWP vector, statistic and uncertainty remain unverified. | Unresolved source identity; do not count. |
| Lee & Tzeng 2008 | The [official ASCE article record](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9402%282008%29134%3A2%2853%29) defines the functional unit as generated electricity and covers turbine manufacturing, foundation construction, operation and disposal. Its abstract reports CO2, not a full multi-gas GHG inventory; it gives no GWP horizon or uncertainty statistic. It acknowledges an earlier 2006 EcoBalance version. Its references include other wind studies, but no shared-inventory dependence is directly established by the abstract. | CO2-only case; fails full-GHG common scope; no group admitted. |
| Rule, Worth & Boyle 2009 | The [official PubMed record](https://pubmed.ncbi.nlm.nih.gov/19746744/) describes an LCA of a New Zealand wind-farm case from construction through decommissioning, including maintenance. The reported indicator is CO2 per generated kWh, not a verified complete GHG vector; no GWP horizon or uncertainty distribution/interval is given in the abstract. No shared inventory was established. The [ACS publisher record](https://pubs.acs.org/doi/10.1021/es900125e) is access restricted. | CO2-only case; fails full-GHG common scope; no group admitted. |

This batch adds **zero independent common-scope groups**. Enel lacks accessible
primary boundary/gas/statistic evidence; Khan studies an integrated hybrid;
Lee and Tzeng and Rule report CO2-only summaries; and Krewitt's source identity
is unresolved. The batch does not establish whether 20 compatible independent
groups exist. No model or held-out data were accessed or changed; no emissions
values were transcribed or imported.

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
