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
The oversegmentation by kmeans uses a binned meshgrid to search on as this both forces locality on the segmentations of the system and dramatically speeds up the algorithm when varying the number of segments (double the kmeans centers only increase the runtime by ~15%). This is a key part to the SLIC algorithm. 

Although rough locality is forced there is nothing ensuring segments are fully connected, which is particularly significant for rock grains with spot like patters as they are often picked out in xyrgb space to be in a neighbouring segement. As such a method for checking connectivity and dealing with it appopirately is needed.

Recombination metrics have been chosen by . . . . . .

## Code structure
The kmeans code is built on an inherited class system; where by the bin mesh, distance metrics and image feature conversion are handeled by parent classes. This means that the kmeans_local class will not need changing to work for more complicated versions of the algorithms.

The recombination code structure works by making segment class instances which identify their edges, which neighbours they are with and then calculate metrics by which to combine. To make the process of combination not too complex, the whole image is scaned then all segments to merge are merged, preventing the need for figuring out which segments neighbours have changed, etc. After the first scan segments instances are reinitalised which is inefficent in terms of overhead but allows for the recycling of code for a more elegent overall structure. Since the number of segments in this algorithm is always going to small compared to the kmeans seperation method the optimisation of this algorithm is less significant, however if the run time becomes an issue this will be readdressed.

## Notes to the Examiner
enjoy :)
