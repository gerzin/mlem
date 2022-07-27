# BlackBoxes

## noclustering

Clustering effettuato solo sul training set per ottenere i centroidi ed estrarre un tot di elementi da ogni quantile.
Il numero alla fine del file contenente i dati indica il numero di elementi estratti da ogni quantile.

* Adult
    * Il training set conteneva 6 cluster
        * provato con 1,2,3 elementi per quantile
* Diva
    * Il training set conteneva 2 cluster
        * provato con 5,7,9 elementi per quantile

Tot elementi dataset di attacco = #cluster * 4 * #elementi-per-quantile

## clustering

Clustering effettuato sulle feature numeriche, ignorando la variabile target.

* Adult
  > k = 6
* Diva
  > k = 2