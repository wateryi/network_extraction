"""
Create on Tuesday, October 22, 2019 10:27:17 PM
By: wangliangju@gmail.com
"""

from os.path import join
import os
import sys
import ntpath
from ntpath import basename
import time
import cv2
import meshpy.triangle as triangle
from skimage.morphology import disk, remove_small_objects, \
    remove_small_holes, binary_opening, binary_closing
import numpy as np
import scipy.misc
from PIL import Image
import networkx as nx

# custom helper functions
import net_helpers as nh


class SegImage:
    def __init__(self, path):
        self.path = path

    def imread(self, flag=cv2.IMREAD_COLOR):
        self.imgraw = cv2.imread(self.path, flag)

    def togray(self):
        if len(self.imgraw.shape) == 2:
            self.imggray = self.imgraw
        elif len(self.imgraw.shape) == 3 and self.imgraw.shape[2] == 3:
            self.imggray = cv2.RGBtoGray(self.imgraw)
        else:
            raise Exception('The image format can not be handled')

    def blur(self, blursize=3):
        blur_kernel = (blursize, blursize)
        self.imggray = cv2.blur(self.imggray, blur_kernel)

    def seg(self, blocksize=51, threshBackgrd=1, inversed=False):
        self.maskBackgrd = np.where(self.imggray < threshBackgrd, 0, 1)
        if self.imggray.dtype == np.uint16:
            gray1 = np.float32(self.imggray)
            gray2 = np.float32(self.imggray)
            gray1[gray1==0] = 65536
            graymin = np.min(gray1)
            graymax = np.max(gray2)
            gray8bit = (np.float32(gray2-graymin)/(graymax-graymin)*254+1)
            gray8bit[gray8bit<0] = 0
            self.imggray = np.uint8(gray8bit)
        if inversed:
            self.imggray = 255 - self.imggray
        image = cv2.adaptiveThreshold(self.imggray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blocksize, 2)
        self.imgbinary = np.where(self.maskBackgrd == 0, 0, image)

    def denoise(self, minimum_feature_size=3000, smoothing=True):
        image = self.imgbinary
        new = remove_small_objects(image.astype(np.bool),
                                   min_size=minimum_feature_size, connectivity=1)
        if new.sum() == 0:
            print('minimum feature size too large, trying again with m = {}'.\
                  format(int(minimum_feature_size/2)))
            new = remove_small_objects(image.astype(np.bool),
                                       min_size=int(minimum_feature_size/2), connectivity=1)
            if new.sum() == 0:
                print('minimum feature size too large, trying again with m = {}'.\
                      format(int(minimum_feature_size/4)))
                new = remove_small_objects(image.astype(np.bool),
                                           min_size=int(minimum_feature_size/4),
                                           connectivity=1)
        image = new

        # smoothe with binary opening and closing
        # standard binary image noise-removal with opening followed by closing
        # maybe remove this processing step if depicted structures are really tiny
        if smoothing:
            image = binary_opening(image, disk(smoothing))
            image = binary_closing(image, disk(smoothing))

        # remove disconnected objects and fill in holes
        image = remove_small_objects(image.astype(bool),
                                     min_size=minimum_feature_size, connectivity=1)
        image = remove_small_holes(image.astype(bool),
                                   min_size=minimum_feature_size/100, connectivity=1)

        self.imgbinary = image


class VeinNet:
    

    def __init__(self, image, imagename='veinnet',
                 dest='./', figure_format='png',
                 dpi=500, graph_format='gpickle',
                 debug=False, verbose=False, plot=False):
        self.pa = "NET>"
        self.imgraw = image
        self.debug = debug
        self.verbose = verbose
        self.imageName = imagename
        self.dest = dest
        self.figure_format = figure_format
        self.dpi = dpi
        self.plot = plot
        self.graph_format = graph_format

        if self.verbose:
            print("\n" + self.pa + "*** Starting to vectorize image ***")
            print("\n" + self.pa + "Current step: Image preparations")
        self.previous_step = time.clock()

    def getDistnaceMap(self, issave=False):
        distance_map = nh.cvDistanceMap(self.imgraw).astype(
            np.int)  # Create distance map
        self.distanceMap = distance_map
        self.height, width = self.distanceMap.shape

        if issave:
            Image.fromarray(distance_map, mode='L')\
                .save(join(self.dest, self.imageName + "_dm.png"), 'PNG')  # save the distance map in case we need it later

        # if debug:
        #     scipy.misc.imsave(join(self.dest, self.imageName + "_processed.png"),image)          #debug output
        if self.verbose:
            step = time.clock()  # progress output
            print(self.pa + "Done in %1.2f sec." % (step-self.previous_step))
            print("\n" + self.pa + "Current step: Contour extraction and thresholding.")
            self.previous_step = step

    def getContours(self):
        """ 
        Contours:
                - Find the contours of the features present in the image.
                The contours are approximated using the Teh-Chin-dominant-point
                detection-algorithm (see Teh, C.H. and Chin, R.T., On the Detection
                of Dominant Pointson Digital Curve. PAMI 11 8, pp 859-872 (1989))
                - Find the longest contour within the set of contours
        """
        image = self.imgraw
        raw_contours = nh.getContours(image)  # Extract raw contours.
        flattened_contours = nh.flattenContours(
            raw_contours)  # Flatten nested contour list
        if self.debug:  # debug output
            print(self.pa + "\tContours converted, we have %i contour(s)."\
                  % (len(flattened_contours)))
        # filter out contours smaller than 3 in case there are any left
        flattened_contours = nh.thresholdContours(flattened_contours, 3)
        if self.debug:  # debug output
            nh.drawContours(self.height, flattened_contours, self.imageName, self.dest,
                            self.figure_format, self.dpi)
        # Find index of longest contour.
        longest_index = 0  # Position of longest contour.
        longest_length = 0  # Length of longest contour.
        # Find index of longest contour.
        for c in range(len(flattened_contours)):
            if(len(flattened_contours[c]) > longest_length):
                longest_length = len(flattened_contours[c])
                longest_index = c

        self.contours = flattened_contours
        self.indLongestContour = longest_index

        if self.verbose:
            step = time.clock()  # progress output
            print(self.pa + "Done in %1.2f sec." % (step-self.previous_step))
            print("\n" + self.pa + "Current step: Contour extraction and thresholding.")
            self.previous_step = step

    def mesh(self):
        """
        Mesh Creation:
                - The mesh is created of points and facets where every facet is the
                plane spanned by one contour.
        """
        flattened_contours = np.asarray(
            self.contours)  # add a bit of noise to increase stability of triangulation algorithm
        longest_index = self.indLongestContour
        for c in flattened_contours:
            for p in c:
                p[0] = p[0] + 0.1*np.random.rand()
                p[1] = p[1] + 0.1*np.random.rand()

        # First add longest contour to mesh.
        mesh_points = flattened_contours[longest_index]
        # Create facets from the longest contour.
        mesh_facets = nh.roundTripConnect(0, len(mesh_points)-1)

        # Every contour other than the longest needs an interiour point.
        hole_points = []
        for i in range(len(flattened_contours)):  # Traverse all contours.
            curr_length = len(mesh_points)
            if(i == longest_index):  # Ignore longest contour.
                pass
            else:  # Find a point that lies within the contour.
                contour = flattened_contours[i]
                interior_point = nh.getInteriorPoint(contour)
                # Add point to list of interior points.
                hole_points.append((interior_point[0], interior_point[1]))
                # Add contours identified by their interior points to the mesh.
                mesh_points.extend(contour)
                mesh_facets.extend(nh.roundTripConnect(
                    curr_length, len(mesh_points)-1))  # Add facets to the mesh

        self.meshPoints = mesh_points
        self.meshFacets = mesh_facets
        self.holePoints = hole_points

        if self.verbose:
            step = time.clock()  # progress output
            print(self.pa + "Done in %1.2f sec." % (step-self.previous_step))
            print("\n" + self.pa + "Current step: meshing.")
            self.previous_step = step

    def triangulate(self):
        """
        Triangulation:
                - set the points we want to triangulate
                - mark the holes we want to ignore by their interior points
                - triangulation: no interior steiner points, we want triangles to fill
                the whole space between two boundaries. Allowing for quality meshing
                would also mess with the triangulation we want.
        """
        mesh_points = self.meshPoints
        hole_points = self.holePoints
        mesh_facets = self.meshFacets
        info = triangle.MeshInfo()  # Create triangulation object.
        info.set_points(mesh_points)  # Set points to be triangulated.
        if(len(hole_points) > 0):
            info.set_holes(hole_points)  # Set holes (contours) to be ignored.
        info.set_facets(mesh_facets)  # Set facets.
        self.triangulation = triangle.build(info, verbose=False, allow_boundary_steiner=False,  # Build Triangulation.
                                            allow_volume_steiner=False, quality_meshing=False)

        if self.verbose:
            step = time.clock()  # progress output
            print(self.pa + "Done in %1.2f sec." % (step-self.previous_step))
            print("\n" + self.pa + "Current step: triangulating.")
            self.previous_step = step

    def classifyTriangle(self):
        """
        Triangle classification:
                - build triangle-objects from the triangulation
                - set the type of each triangle (junction, normal, end or isolated)
                depending on how many neighbors it has
                - set the radius of each triangle by looking up its "midpoint"
                in the distance map
                - get rid of isolated triangles
        """
        triangulation = self.triangulation
        triangles = nh.buildTriangles(triangulation)  # Build triangles
        junction = 0
        normal = 0
        end = 0
        isolated_indices = []
        default_triangles = 0
        for i in range(len(triangles)):
            t = triangles[i]  # set the triangle's type
            t.init_triangle_mesh()
            if t.get_type() == "junction":  # count the number of each triangle type for debugging
                junction += 1
            elif t.get_type() == "normal":
                normal += 1
            elif t.get_type() == "end":
                end += 1
            elif t.get_type() == "isolated":
                isolated_indices.append(i)
        # remove isolated triangles from the list of triangles
        self.triangles = list(
            np.delete(np.asarray(triangles), isolated_indices))
        self.isolated_indices = isolated_indices
        if self.debug:  # debug output
            print(self.pa + "\tTriangle types:")
            print(self.pa + "\tjunction: %d, normal: %d, end: %d, isolated: %d"
                  % (junction, normal, end, len(isolated_indices)))

        if self.verbose:
            step = time.clock()  # progress output
            print(self.pa + "Done in %1.2f sec." % (step-self.previous_step))
            print("\n" + self.pa + "Current step: classify triangles.")
            self.previous_step = step

    def graphy(self):
        """
        Graph creation and improvement
                - prune away the outermost branches to avoid surplus branches due
                to noisy contours
                - create a graph object from the neighborhood relations, coordinates
                and radius stored in the adjacency matrix and
                the list of triangles.
        """
        triangles = self.triangles

        default_triangles = 0
        junction = 0
        normal = 0
        end = 0
        isolated = 0
        for t in triangles:
            t.init_triangle_mesh()
            default_triangles += t.set_center(self.distanceMap)
            if t.get_type() == "junction":  # count the number of each triangle type for debugging
                junction += 1
            elif t.get_type() == "normal":
                normal += 1
            elif t.get_type() == "end":
                end += 1
            elif t.get_type() == "isolated":
                isolated += 1

        if self.debug:
            nh.drawTriangulation(self.height, triangles, self.imageName, self.dest, self.distanceMap,
                                 self.figure_format, self.dpi)  # debug output
            print(self.pa + "\tTriangles defaulted to zero: %d" % default_triangles)
            print(self.pa + "\tTriangle types:")
            print(self.pa + "\tjunction: %d, normal: %d, end: %d, isolated: %d"
                  % (junction, normal, end, len(self.isolated_indices)))

        adjacency_matrix = nh.createTriangleAdjacencyMatrix(triangles)

        if self.verbose:
            step = time.clock()  # progress output
            print(self.pa + "Done in %1.2f sec." % (step-self.previous_step))
            print("\n" + self.pa + "Current step: Graphying.")
            self.previous_step = step

        self.graph = nh.createGraph(adjacency_matrix, triangles, self.height)

    def removeRedundantNode(self, node_size=4, redundancy=0, order=5):
        """
        Redundant node removal
                - if so specified, remove half the redundant nodes (i.e. nodes with
                degree 2), draw and save the graph
                - if so specified, remove all the redundant nodes, draw and save the 
                graph
        """
        parameters = {'r': redundancy, 'p': order}
        G = self.graph
        if redundancy == 2:
            nh.drawAndSafe(G, self.imageName, self.dest, parameters, self.verbose,
                           self.plot, self.figure_format, self.dpi,
                           self.graph_format, node_size)  # draw and safe graph with redundant nodes
        if self.debug:
            nh.drawGraphTriangulation(self.height, G, self.triangles, self.imageName, self.dest,
                                      self.distanceMap, self.figure_format, self.dpi)

        if redundancy == 1:  # draw and safe graph with half redundant nodes
            G = nh.removeRedundantNodes(G, self.verbose, 1)
            nh.drawAndSafe(G, self.imageName, self.dest, parameters, self.verbose,
                           self.plot, self.figure_format, self.dpi,
                           self.graph_format, node_size)

        if redundancy == 0:
            # draw and safe graph without redundant nodes
            G = nh.removeRedundantNodes(G, self.verbose, 0)
            nh.drawAndSafe(G, self.imageName, self.dest, parameters, self.verbose,
                           self.plot, self.figure_format, self.dpi,
                           self.graph_format, node_size)

        if self.verbose:
            step = time.clock()  # progress output
            print(self.pa + "Done in %1.2f sec." % (step-self.previous_step))
            print("\n" + self.pa + "Current step: Removing redundant nodes.")
            self.previous_step = step

        self.graph = G


class VeinPara:
    def __init__(self, g):
        self.graph = g

    def NumberOfJunctions(self):
        G = self.graph
        junctions = 0
        for n in G.nodes():
            if type(G) == type(nx.DiGraph):
                if len(list(G.neighbors(n))) >= 2:
                    junctions += 1
            else:
                if len(list(G.neighbors(n))) >= 3:
                    junctions += 1

        return junctions

    def NumberOfTips(self):
        G = self.graph
        tips = 0
        for n in G.nodes():
            if type(G) == type(nx.DiGraph):
                if len(list(G.neighbors(n))) == 0:
                    tips += 1
            else:
                if len(list(G.neighbors(n))) == 1:
                    tips += 1

        return tips

    def TotalLength(self):
        G = self.graph
        return np.asarray([e[2]['weight'] for e in list(G.edges(data=True))]).sum()

    def AverageEdgeLength(self):
        G = self.graph
        return np.asarray([e[2]['weight'] for e in list(G.edges(data=True))]).mean()

    def AverageEdgeRadius(self):
        G = self.graph
        return np.asarray([e[2]['conductivity'] for e in list(G.edges(data=True))]).mean()

    def TotalNetworkArea(self):
        G = self.graph
        return np.asarray([e[2]['weight']*e[2]['conductivity']
                           for e in list(G.edges(data=True))]).sum()

    def AreaOfConvexHull(self):
        G = self.graph
        points = np.asarray([[n[1]['y'], n[1]['x']]
                             for n in G.nodes(data=True)])
        hull = scipy.spatial.ConvexHull(points)
        vertices = points[hull.vertices]
        vertices = np.vstack([vertices, vertices[0, 0:]])
        lines = np.hstack([vertices, np.roll(vertices, -1, axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1, y1, x2, y2 in lines))
        return area

    def NumberOfCycles(self):
        G = self.graph
        return len(nx.cycle_basis(G))

    def allParas(self):
        paradict = {'NumberOfJunctions': self.NumberOfJunctions(),
                    'NumberOfTips': self.NumberOfTips(),
                    'NumberOfCycles': self.NumberOfCycles(),
                    'TotalLength': self.TotalLength(),
                    'TotalNetworkArea': self.TotalNetworkArea(),
                    'AreaOfConvexHull': self.AreaOfConvexHull(),
                    'AverageEdgeLength': self.AverageEdgeLength(),
                    'AverageEdgeRadius': self.AverageEdgeRadius()}
        return paradict
