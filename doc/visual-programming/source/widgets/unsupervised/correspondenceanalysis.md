Correspondence Analysis
=======================

Correspondence analysis for categorical multivariate data.

**Inputs**

- Data: input dataset

- Contingency: A precomputed contingency input (if present it takes
  precedence over the **Data** input).

**Outputs**

- Coordinates: coordinates of all components

[Correspondence Analysis](https://en.wikipedia.org/wiki/Correspondence_analysis)
(CA) is a exploratory technique designed to analyze two-way contingency tables.

This widget implements both Simple CA and Multiple Correspondence Analysis
([1]_). Furthermore the Simple CA can be applied to a *concatenated* table
composed of several two-way cross tabulations ([GreenacreBPP9]_,
[GreenacreCAP17]_))

The method is selected in the top of the left widget.

![](images/CorrespondenceAnalysis-stamped.png)

1. Select the method type (simple or multiple CA)


Correspondence Analysis with stacked Tables
-------------------------------------------

![](images/CorrespondenceAnalysis.png)

1. Select the variables for analysis. Multiple variables can be selected for
   analysis and assigned to either Row or Column role. Select the variables
   in the view and right click to bring up the context menu, or double click a
   single row to start inline editing. Assigning one variable to 'Row role'
   and one to 'Column role' performs the simplest form of CA. Assigning
   multiple variables to either roles performs CA on a concatenated table (see
   `Example 2`_). For instance on zoo dataset, selecting 'hair' and 'tail'
   as the 'Columns' and type as the Row would cross tabulate

```
   =========== ==== ==== ==== ====
   \              hair     tail
   ----------- --------- ---------
   type          0    1    0    1
   =========== ==== ==== ==== ====
   amphibian     4    0    3    0
   bird         20    0    0   20
   ...
   reptile       5    0    0    5
   =========== ==== ==== ==== ====
```


2. Select the principal dimensions to plot. The contributions to the total
   inertia are displayed to the left including the total percentage of
   the inertia ([2]_) explained by the two selected dimensions.

3. Select the plot type:

   * Symmetric - Both the row and column points are plotted in their
     respective principal coordinates.

   * Row principal - The row points are plotted in the principal coordinates;
     while column points are in standard coordinates.

   * Column principal - The column points are plotted in the principal
     coordinates; row points in principal coordinates.

   The asymmetric plots are bi-plots. The standard coordinates are the
   'axes' that span the row/column principal coordinates (i.e the row
   principal coordinates are a convex linear combinations of column standard
   coordinates and vice-versa).

   The 'Display standard coords. as arrows' switches the display of standard
   coordinates between point or arrow display.

4. Select the size for the column/row points.

   * Same - All points have the same size.
   * Mass - Points are sized relative to their mass (row/column sum in
     the contingency matrix).
   * Inertia - Points are sized relative to their contribution to the explained
     inertia in the *displayed* dimensions.
   * Inertia relative  - Points are sized relative to their contribution to
     the explained inertia over *all* the dimensions.

5. Produce a report.


If the widget has 'Contingency' input then the view switches to the
*Simple Correspondence Analysis* and the contingency is fixed (the variable
selection view on the left is disabled).


Example 1
---------

![](images/CorrespondenceAnalysis-Smokers.png)

Below, is a simple use of **Correspondence Analysis** on the *smokers_ct*
dataset. The smoking dataset is a synthetic dataset with two categorical
variables recording the smoking habits (none, low, medium, high) of staff
groups (junior/senior managers/employees and secretaries).


Example 2
---------

.. figure:: imagers/CorrespondenceAnalysis-WG93.png

We use ISSS - 1993 survey of ... to science where responents were asked a
number if questions regarding their attitude toward science (column A-G with
5 levels each ranked from strongly disagree,... strongly agree). Along side
the the demographic variables sex (0=male, 1=female), age (0-6)
education level(0-6)


.. [GreenacreBPP9] `Michael Greenacre - Biplots in Practice`_ - Chapter 9

.. [GreenacreBPP10] `Michael Greenacre - Biplots in Practice`_ - Chapter 10

.. [GreenacreCAP17] Michael Greenacre - Correspondence Analysis in Practice -
   Chapter 17


References
----------

[Michael Greenacre - Biplots in Practice](https://www.fbbva.es/microsite/multivariate-statistics/biplots.html)


.. [1] MCA is implemented us using the burt table method with inertia
   adjustment [GreenacreBPP10]_.

.. [2] Inertia is defined as the value of the  |chisq| statistic on the
   contingency table divided by the total count.


.. |chisq| unicode:: U+03C7 U+00B2
