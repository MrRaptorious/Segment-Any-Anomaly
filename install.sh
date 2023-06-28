# $ProjectRoot: the root you save our project, e.g., /home/anyad/VAND-solution
ProjectRoot=<Your-Workspace-Path-here!>
cd $ProjectRoot

# weights
mkdir weights
cd ./weights/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth