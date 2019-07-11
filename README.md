# Automated Thin Section Analysis
#### MSc Applied Computational Science and Engineering
#### Independent Research Project
#### Richard Boyne 
Github: Boyne272

CID: 01057503

## Repository Structure
- images/ holds a few sample images which are cropped versions of a whole thin section images. The main images used in this project are not stored in this repo for data rights reasons.
- kmeans/ holds the vairious versions of the oversegmentation methods by kmeans
- "Peliminary Report Richard Boyne" is the project plan, submitted on the 5th of June 2019


## Implementation notes
The oversegmentation by kmeans uses a binned meshgrid to search on as this both forces locality on the segmentations of the system and dramatically speeds up the algorithm when varying the number of segments (double the kmeans centers only increase the runtime by ~15%).


## Code structure
The kmeans code is built on an inherited class system; where by the bin mesh, distance metrics and image feature conversion are handeled by parent classes. This means that the kmeans_local class will not need changing to work for more complicated versions of the algorithms.


## Notes to the Examiner
enjoy :)
