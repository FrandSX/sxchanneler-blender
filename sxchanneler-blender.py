bl_info = {
    'name': 'SX Channeler',
    'author': 'Jani Kahrama / Secret Exit Ltd.',
    'version': (0, 0, 1),
    'blender': (3, 5, 0),
    'location': 'View3D',
    'description': 'Channel Copy Tool',
    'doc_url': '',
    'tracker_url': 'https://github.com/FrandSX/sxchanneler-blender/issues',
    'category': 'Development',
}

import bpy


# Global variable to store the data
stored_data = {
    'colors': {},
    'uvs': {}
}


# ------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------
class SXTOOLS2_sxglobals(object):
    def __init__(self):
        self.copy_buffer = {}

    def __del__(self):
        print('SX Tools: Exiting sxglobals')


# ------------------------------------------------------------------------
#    Color Conversions
# ------------------------------------------------------------------------
class SXTOOLS2_convert(object):
    def __init__(self):
        return None


    def color_to_luminance(self, in_rgba, premul=True):
        lumR = 0.212655
        lumG = 0.715158
        lumB = 0.072187
        alpha = in_rgba[3]

        linLum = lumR * in_rgba[0] + lumG * in_rgba[1] + lumB * in_rgba[2]
        # luminance = convert.linear_to_srgb((linLum, linLum, linLum, alpha))[0]
        if premul:
            return linLum * alpha  # luminance * alpha
        else:
            return linLum


    def luminance_to_color(self, value):
        return (value, value, value, 1.0)


    def luminance_to_alpha(self, value):
        return (1.0, 1.0, 1.0, value)


    def srgb_to_linear(self, in_rgba):
        out_rgba = []
        for i in range(3):
            if in_rgba[i] < 0.0:
                out_rgba.append(0.0)
            elif 0.0 <= in_rgba[i] <= 0.0404482362771082:
                out_rgba.append(float(in_rgba[i]) / 12.92)
            elif 0.0404482362771082 < in_rgba[i] <= 1.0:
                out_rgba.append(((in_rgba[i] + 0.055) / 1.055) ** 2.4)
            elif in_rgba[i] > 1.0:
                out_rgba.append(1.0)
        out_rgba.append(in_rgba[3])
        return out_rgba


    def linear_to_srgb(self, in_rgba):
        out_rgba = []
        for i in range(3):
            if in_rgba[i] < 0.0:
                out_rgba.append(0.0)
            elif 0.0 <= in_rgba[i] <= 0.00313066844250063:
                out_rgba.append(float(in_rgba[i]) * 12.92)
            elif 0.00313066844250063 < in_rgba[i] <= 1.0:
                out_rgba.append(1.055 * in_rgba[i] ** (float(1.0)/2.4) - 0.055)
            elif in_rgba[i] > 1.0:
                out_rgba.append(1.0)
        out_rgba.append(in_rgba[3])
        return out_rgba


    def rgb_to_hsl(self, in_rgba):
        R, G, B = in_rgba[:3]
        Cmax = max(R, G, B)
        Cmin = min(R, G, B)

        H = 0.0
        S = 0.0
        L = (Cmax+Cmin)/2.0

        if L == 1.0:
            S = 0.0
        elif 0.0 < L < 0.5:
            S = (Cmax-Cmin)/(Cmax+Cmin)
        elif L >= 0.5:
            S = (Cmax-Cmin)/(2.0-Cmax-Cmin)

        if S > 0.0:
            if R == Cmax:
                H = ((G-B)/(Cmax-Cmin))*60.0
            elif G == Cmax:
                H = ((B-R)/(Cmax-Cmin)+2.0)*60.0
            elif B == Cmax:
                H = ((R-G)/(Cmax-Cmin)+4.0)*60.0

        return [H/360.0, S, L]


    def hsl_to_rgb(self, in_value):
        H, S, L = in_value

        v1 = 0.0
        v2 = 0.0

        rgb = [0.0, 0.0, 0.0]

        if S == 0.0:
            rgb = [L, L, L]
        else:
            if L < 0.5:
                v1 = L*(S+1.0)
            elif L >= 0.5:
                v1 = L+S-L*S

            v2 = 2.0*L-v1

            # H = H/360.0

            tR = H + 0.333333
            tG = H
            tB = H - 0.333333

            tList = [tR, tG, tB]

            for i, t in enumerate(tList):
                if t < 0.0:
                    t += 1.0
                elif t > 1.0:
                    t -= 1.0

                if t*6.0 < 1.0:
                    rgb[i] = v2+(v1-v2)*6.0*t
                elif t*2.0 < 1.0:
                    rgb[i] = v1
                elif t*3.0 < 2.0:
                    rgb[i] = v2+(v1-v2)*(0.666666 - t)*6.0
                else:
                    rgb[i] = v2

        return rgb


    def colors_to_values(self, colors, as_rgba=False):
        count = len(colors) // 4
        if as_rgba:
            for i in range(count):
                color = colors[(0+i*4):(4+i*4)]
                lum = self.color_to_luminance(color, premul=False)
                colors[(0+i*4):(4+i*4)] = [lum, lum, lum, color[3]]
            return colors
        else:
            values = [None] * count
            for i in range(count):
                values[i] = self.color_to_luminance(colors[(0+i*4):(4+i*4)])
            return values


    def values_to_colors(self, values, invert=False, as_alpha=False, with_mask=False, as_tuple=False):
        if invert:
            values = self.invert_values(values)

        count = len(values) 
        colors = [None] * count * 4
        for i in range(count):
            if as_alpha:
                colors[(0+i*4):(4+i*4)] = [1.0, 1.0, 1.0, values[i]]
            elif with_mask:
                alpha = 1.0 if values[i] > 0.0 else 0.0
                colors[(0+i*4):(4+i*4)] = [values[i], values[i], values[i], alpha]
            else:
                colors[(0+i*4):(4+i*4)] = self.luminance_to_color(values[i])

        if as_tuple:
            rgba = [None] * count
            for i in range(count):
                rgba[i] = tuple(colors[(0+i*4):(4+i*4)])
            return rgba
        else:
            return colors


    def invert_values(self, values):
        return [1.0 - value for value in values]


    def __del__(self):
        print('SX Tools: Exiting convert')



# ------------------------------------------------------------------------
#    Value Generators and Utils
#    NOTE: Switching between EDIT and OBJECT modes is slow.
#          Make sure OBJECT mode is enabled before calling
#          any functions in this class!
# ------------------------------------------------------------------------
class SXTOOLS2_generate(object):
    def __init__(self):
        return None


    def blur_list(self, obj, layer, masklayer=None, returndict=False):

        def average_color(colors):
            sum_color = Vector((0.0, 0.0, 0.0, 0.0))
            for color in colors:
                sum_color += color
            return (sum_color / len(colors))


        color_dict = {}
        vert_blur_dict = {}
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.types.BMVertSeq.ensure_lookup_table(bm.verts)

        colors = layers.get_layer(obj, layer)
        for vert in bm.verts:
            loop_colors = []
            for loop in vert.link_loops:
                loop_color = Vector(colors[(loop.index*4):(loop.index*4+4)])
                loop_colors.append(loop_color)
            
            color_dict[vert.index] = average_color(loop_colors)

        for vert in bm.verts:
            num_connected = len(vert.link_edges)
            if num_connected > 0:
                edge_weights = [(edge.other_vert(vert).co - vert.co).length for edge in vert.link_edges]
                max_weight = max(edge_weights)

                # weights are inverted so near colors are more important
                edge_weights = [int((1.1 - (weight / max_weight)) * 10) for weight in edge_weights]

                neighbor_colors = [Vector(color_dict[vert.index])] * 20
                for i, edge in enumerate(vert.link_edges):
                    for j in range(edge_weights[i]):
                        neighbor_colors.append(Vector(color_dict[edge.other_vert(vert).index]))

                vert_blur_dict[vert.index] = average_color(neighbor_colors)
            else:
                vert_blur_dict[vert.index] = color_dict[vert.index]

        bm.free()

        if returndict:
            return vert_blur_dict
        else:
            vert_blur_list = self.vert_dict_to_loop_list(obj, vert_blur_dict, 4, 4)
            blur_list = self.mask_list(obj, vert_blur_list, masklayer)

            return blur_list


    def curvature_list(self, obj, norm_objs, masklayer=None, returndict=False):

        def generate_curvature_dict(curv_obj):
            vert_curv_dict = {}
            mesh = curv_obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.normal_update()

            # pass 1: calculate curvatures
            for vert in bm.verts:
                numConnected = len(vert.link_edges)
                if numConnected > 0:
                    edgeWeights = []
                    angles = []
                    for edge in vert.link_edges:
                        edgeWeights.append(edge.calc_length())
                        angles.append(math.acos(vert.normal.normalized() @ (edge.other_vert(vert).co - vert.co).normalized()))

                    total_weight = sum(edgeWeights)

                    vtxCurvature = 0.0
                    for i in range(numConnected):
                        curvature = angles[i] / math.pi - 0.5
                        vtxCurvature += curvature
                        # weighted_curvature = (curvature * edgeWeights[i]) / total_weight
                        # vtxCurvature += weighted_curvature

                    vtxCurvature = min(vtxCurvature / float(numConnected), 1.0)

                    vert_curv_dict[vert.index] = round(vtxCurvature, 5)
                else:
                    vert_curv_dict[vert.index] = 0.0

            # pass 2: if tiling, copy the curvature value from the connected vert of the edge that's pointing away from the mirror axis
            if curv_obj.sx2.tiling:
                if 'sxTiler' in curv_obj.modifiers:
                    curv_obj.modifiers['sxTiler'].show_viewport = False
                    curv_obj.modifiers.update()
                    bpy.context.view_layer.update()

                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([curv_obj, ], local=True)
                tiling_props = [('tile_neg_x', 'tile_pos_x'), ('tile_neg_y', 'tile_pos_y'), ('tile_neg_z', 'tile_pos_z')]
                axis_vectors = [(Vector((-1.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0))), (Vector((0.0, -1.0, 0.0)), Vector((0.0, 1.0, 0.0))), (Vector((0.0, 0.0, -1.0)), Vector((0.0, 0.0, 1.0)))]
                bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

                for vert in bm.verts:
                    bound_ids = None
                    for i, coord in enumerate(vert.co):
                        for j, prop in enumerate(tiling_props[i]):
                            if getattr(obj.sx2, prop) and (round(coord, 2) == round(bounds[i][j], 2)):
                                bound_ids = (i, j)

                    if bound_ids is not None:
                        numConnected = len(vert.link_edges)
                        if numConnected > 0:
                            angles = []
                            other_verts = []
                            for i, edge in enumerate(vert.link_edges):
                                edgeVec = edge.other_vert(vert).co - vert.co
                                angles.append(axis_vectors[bound_ids[0]][bound_ids[1]].dot(edgeVec))
                                other_verts.append(edge.other_vert(vert))
                            value_vert_id = other_verts[angles.index(min(angles))].index
                            vert_curv_dict[vert.index] = vert_curv_dict[value_vert_id]
                        else:
                            vert_curv_dict[vert.index] = 0.0

                if 'sxTiler' in curv_obj.modifiers:
                    curv_obj.modifiers['sxTiler'].show_viewport = curv_obj.sx2.tile_preview
                    curv_obj.modifiers.update()
                    bpy.context.view_layer.update()

            bm.free()
            return vert_curv_dict


        scene = bpy.context.scene.sx2
        normalizeconvex = scene.normalizeconvex
        normalizeconcave = scene.normalizeconcave
        invert = scene.toolinvert

        # Generate curvature dictionary of all objects in a multi-selection
        if normalizeconvex or normalizeconcave:
            norm_obj_curvature_dict = {norm_obj: generate_curvature_dict(norm_obj) for norm_obj in norm_objs}
            min_curv = min(min(curv_dict.values()) for curv_dict in norm_obj_curvature_dict.values())
            max_curv = max(max(curv_dict.values()) for curv_dict in norm_obj_curvature_dict.values())
            vert_curv_dict = norm_obj_curvature_dict[obj]

            # Normalize convex and concave separately
            # to maximize artist ability to crease
            for vert, vtxCurvature in vert_curv_dict.items():
                if (vtxCurvature < 0.0) and normalizeconcave:
                    vert_curv_dict[vert] = (vtxCurvature / float(min_curv)) * -0.5 + 0.5
                elif (vtxCurvature > 0.0) and normalizeconvex:
                    vert_curv_dict[vert] = (vtxCurvature / float(max_curv)) * 0.5 + 0.5
                else:
                    vert_curv_dict[vert] = (vtxCurvature + 0.5)
        else:
            vert_curv_dict = generate_curvature_dict(obj)
            for vert, vtxCurvature in vert_curv_dict.items():
                vert_curv_dict[vert] = (vtxCurvature + 0.5)

        if invert:
            for key, value in vert_curv_dict.items():
                vert_curv_dict[key] = 1.0 - value

        if returndict:
            return vert_curv_dict

        else:
            vert_curv_list = self.vert_dict_to_loop_list(obj, vert_curv_dict, 1, 4)
            curv_list = self.mask_list(obj, vert_curv_list, masklayer)

            return curv_list


    def direction_list(self, obj, masklayer=None):
        scene = bpy.context.scene.sx2
        cone_angle = scene.dirCone
        half_cone_angle = cone_angle * 0.5

        vert_dict = self.vertex_data_dict(obj, masklayer)

        if vert_dict:
            vert_dir_dict = {vert_id: 0.0 for vert_id in vert_dict}
            inclination = math.radians(scene.dirInclination - 90.0)
            angle = math.radians(scene.dirAngle + 90)
            direction = Vector((math.sin(inclination) * math.cos(angle), math.sin(inclination) * math.sin(angle), math.cos(inclination)))

            for vert_id in vert_dict:
                vert_world_normal = Vector(vert_dict[vert_id][3])
                angle_diff = math.degrees(math.acos(min(1.0, max(-1.0, vert_world_normal @ direction))))
                vert_dir_dict[vert_id] = max(0.0, (90.0 + half_cone_angle - angle_diff) / (90.0 + half_cone_angle))

            values = self.vert_dict_to_loop_list(obj, vert_dir_dict, 1, 1)
            vert_dir_list = [None] * len(values) * 4
            for i in range(len(values)):
                vert_dir_list[(0+i*4):(4+i*4)] = [values[i], values[i], values[i], 1.0]

            return self.mask_list(obj, vert_dir_list, masklayer)
        else:
            return None


    def noise_list(self, obj, amplitude=0.5, offset=0.5, mono=False, masklayer=None):

        def make_noise(amplitude, offset, mono):
            col = [None, None, None, 1.0]
            if mono:
                monoval = offset+random.uniform(-amplitude, amplitude)
                for i in range(3):
                    col[i] = monoval
            else:
                for i in range(3):
                    col[i] = offset+random.uniform(-amplitude, amplitude)
            return col

        random.seed(sxglobals.randomseed)
        vert_ids = self.vertex_id_list(obj)
        noise_dict = {vtx_id: make_noise(amplitude, offset, mono) for vtx_id in vert_ids}
        noise_list = self.vert_dict_to_loop_list(obj, noise_dict, 4, 4)
        return self.mask_list(obj, noise_list, masklayer)


    def ray_randomizer(self, count):
        hemisphere = [None] * count
        random.seed(sxglobals.randomseed)

        for i in range(count):
            u1 = random.random()
            u2 = random.random()
            r = math.sqrt(u1)
            theta = 2*math.pi*u2

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = math.sqrt(max(0, 1 - u1))

            ray = Vector((x, y, z))
            up_vector = Vector((0, 0, 1))
            
            dot_product = ray.dot(up_vector)
            hemisphere[i] = (ray, dot_product)

        sorted_hemisphere = sorted(hemisphere, key=lambda x: x[1], reverse=True)
        return sorted_hemisphere


    def ground_plane(self, size, pos):
        size *= 0.5
        vert_list = [
            (pos[0]-size,pos[1]-size, pos[2]),
            (pos[0]+size, pos[1]-size, pos[2]),
            (pos[0]-size, pos[1]+size, pos[2]),
            (pos[0]+size, pos[1]+size, pos[2])]
        face_list = [(0, 1, 3, 2)]

        mesh = bpy.data.meshes.new('groundPlane_mesh')
        groundPlane = bpy.data.objects.new('groundPlane', mesh)
        bpy.context.scene.collection.objects.link(groundPlane)

        mesh.from_pydata(vert_list, [], face_list)
        mesh.update(calc_edges=True)

        # groundPlane.location = pos
        return groundPlane, mesh


    def thickness_list(self, obj, raycount, masklayer=None):

        def dist_caster(obj, vert_dict):
            hemisphere = self.ray_randomizer(20)
            for vert_id, vert_data in vert_dict.items():
                vertLoc = vert_data[0]
                vertNormal = vert_data[1]
                invNormal = -vertNormal
                rotQuat = forward.rotation_difference(invNormal)
                bias = 0.001

                # Raycast for bias
                hit, loc, normal, _ = obj.ray_cast(vertLoc, invNormal)
                if hit and (normal.dot(invNormal) < 0):
                    hit_dist = (loc - vertLoc).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                # offset ray origin with normal bias
                vertPos = vertLoc + (bias * invNormal)

                # Raycast for distance
                for ray, _ in hemisphere:
                    hit, loc, normal, _ = obj.ray_cast(vertPos, rotQuat @ Vector(ray))
                    if hit:
                        dist_list.append((loc - vertPos).length)

                bias_vert_dict[vert_id] = (vertPos, invNormal)


        def sample_caster(obj, raycount, vert_dict, raydistance=1.70141e+38):
            hemisphere = self.ray_randomizer(raycount)
            for vert_id, vert_data in vert_dict.items():
                vertPos = vert_data[0]
                invNormal = vert_data[1]
                rotQuat = forward.rotation_difference(invNormal)

                for ray, _ in hemisphere:
                    hit = obj.ray_cast(vertPos, rotQuat @ Vector(ray), distance=raydistance)[0]
                    vert_occ_dict[vert_id] += contribution * hit


        vert_dict = self.vertex_data_dict(obj, masklayer)
        if vert_dict:
            edg = bpy.context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(edg)
            contribution = 1.0 / float(raycount)
            forward = Vector((0.0, 0.0, 1.0))
            dist_list = []
            bias_vert_dict = {}
            vert_occ_dict = {vert_id: 0.0 for vert_id in vert_dict}

            # Pass 1: analyze ray hit distances, set max ray distance to half of median distance
            dist_caster(obj_eval, vert_dict)
            distance = statistics.median(dist_list) * 0.5

            # Pass 2: final results
            sample_caster(obj_eval, raycount, bias_vert_dict, raydistance=distance)

            vert_occ_list = generate.vert_dict_to_loop_list(obj, vert_occ_dict, 1, 4)
            return self.mask_list(obj, vert_occ_list, masklayer)
        else:
            return None


    def occlusion_list(self, obj, raycount=250, blend=0.5, dist=10.0, groundplane=False, groundheight=-0.5, masklayer=None):
        # start_time = time.time()

        scene = bpy.context.scene
        contribution = 1.0/float(raycount)
        hemisphere = self.ray_randomizer(raycount)
        mix = max(min(blend, 1.0), 0.0)
        forward = Vector((0.0, 0.0, 1.0))

        if obj.sx2.tiling:
            blend = 0.0
            groundplane = False
            if not 'sxTiler' in obj.modifiers:
                modifiers.add_modifiers([obj, ])
            obj.modifiers['sxTiler'].show_viewport = False
            bpy.context.view_layer.update()
            xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([obj, ], local=True)
            dist = 2.0 * min(xmax-xmin, ymax-ymin, zmax-zmin)
            obj.modifiers['sxTiler'].show_viewport = True
            bpy.context.view_layer.update()

        vert_occ_dict = {}
        vert_dict = self.vertex_data_dict(obj, masklayer, dots=True)

        if vert_dict:

            if (blend > 0.0) and groundplane:
                pivot = utils.find_root_pivot([obj, ])
                pivot = (pivot[0], pivot[1], groundheight)
                size = max(obj.dimensions) * 10
                ground, groundmesh = self.ground_plane(size, pivot)

            edg = bpy.context.evaluated_depsgraph_get()
            # edg.update()
            # obj_eval = obj.evaluated_get(edg)
            bvh = BVHTree.FromObject(obj, edg)

            for vert_id in vert_dict:
                bias = 0.001
                occValue = 1.0
                scnOccValue = 1.0
                vertLoc, vertNormal, vertWorldLoc, vertWorldNormal, min_dot = vert_dict[vert_id]

                # use modified tile-border normals to reduce seam artifacts
                # if vertex pos x y z is at bbx limit, and mirror axis is set, modify respective normal vector component to zero
                if obj.sx2.tiling:
                    mod_normal = list(vertNormal)
                    match = False

                    tiling_props = [('tile_neg_x', 'tile_pos_x'), ('tile_neg_y', 'tile_pos_y'), ('tile_neg_z', 'tile_pos_z')]
                    bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

                    for i, coord in enumerate(vertLoc):
                        for j, prop in enumerate(tiling_props[i]):
                            if getattr(obj.sx2, prop) and (round(coord, 2) == round(bounds[i][j], 2)):
                                match = True
                                mod_normal[i] = 0.0

                    if match:
                        vertNormal = Vector(mod_normal[:]).normalized()

                # Pass 0: Raycast for bias
                # hit, loc, normal, _ = obj.ray_cast(vertLoc, vertNormal, distance=dist)
                result = bvh.ray_cast(vertLoc, vertNormal, dist)
                hit = not all(x is None for x in result)
                _, normal, _, hit_dist = result
                if hit and (normal.dot(vertNormal) > 0):
                    # hit_dist = (loc - vertLoc).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                # Pass 1: Mark hits for rays that are inside the mesh
                first_hit_index = raycount
                for i, (_, dot) in enumerate(hemisphere):
                    if dot < min_dot:
                        first_hit_index = i
                        break

                valid_rays = [ray for ray, _ in hemisphere[:first_hit_index]]
                occValue -= contribution * (raycount - first_hit_index)

                # Store Pass 2 valid ray hits
                pass2_hits = [False] * len(valid_rays)

                # Pass 2: Local space occlusion for individual object
                if 0.0 <= mix < 1.0:
                    rotQuat = forward.rotation_difference(vertNormal)

                    # offset ray origin with normal bias
                    vertPos = vertLoc + (bias * vertNormal)

                    # for every object ray hit, subtract a fraction from the vertex brightness
                    for i, ray in enumerate(valid_rays):
                        # hit = obj_eval.ray_cast(vertPos, rotQuat @ Vector(ray), distance=dist)[0]
                        result = bvh.ray_cast(vertPos, rotQuat @ Vector(ray), dist)
                        hit = not all(x is None for x in result)
                        occValue -= contribution * hit
                        pass2_hits[i] = hit

                # Pass 3: Worldspace occlusion for scene
                if 0.0 < mix <= 1.0:
                    rotQuat = forward.rotation_difference(vertWorldNormal)

                    # offset ray origin with normal bias
                    scnVertPos = vertWorldLoc + (bias * vertWorldNormal)

                    # Include previous pass results
                    scnOccValue = occValue

                    # Fire rays only for samples that had not hit in Pass 2
                    for i, ray in enumerate(valid_rays):
                        if not pass2_hits[i]:
                            hit = scene.ray_cast(edg, scnVertPos, rotQuat @ Vector(ray), distance=dist)[0]
                            scnOccValue -= contribution * hit

                vert_occ_dict[vert_id] = float((occValue * (1.0 - mix)) + (scnOccValue * mix))

            if (blend > 0.0) and groundplane:
                bpy.data.objects.remove(ground, do_unlink=True)
                bpy.data.meshes.remove(groundmesh, do_unlink=True)

            if obj.sx2.tiling:
                obj.modifiers['sxTiler'].show_viewport = False

            vert_occ_list = generate.vert_dict_to_loop_list(obj, vert_occ_dict, 1, 4)
            result = self.mask_list(obj, vert_occ_list, masklayer)

            # end_time = time.time()  # Stop the timer
            # print("SX Tools: AO rendered in {:.4f} seconds".format(end_time - start_time)) 

            return result

        else:
            return None


    def emission_list(self, obj, raycount=250, masklayer=None):

        def calculate_face_colors(obj):
            colors = layers.get_layer(obj, obj.sx2layers['Emission'], as_tuple=True)
            face_colors = [None] * len(obj.data.polygons)

            for face in obj.data.polygons:
                face_color = Vector((0.0, 0.0, 0.0, 0.0))
                for loop_index in face.loop_indices:
                    loop_color = Vector(colors[loop_index])
                    if loop_color[3] > 0.0:
                        face_color += Vector(colors[loop_index])
                face_colors[face.index] = face_color / len(face.loop_indices)

            return face_colors


        def face_colors_to_loop_list(obj, face_colors):
            loop_list = self.empty_list(obj, 4)

            for poly in obj.data.polygons:
                for loop_idx in poly.loop_indices:
                    loop_list[(0+loop_idx*4):(4+loop_idx*4)] = face_colors[poly.index]

            return loop_list


        _, empty = layers.get_layer_mask(obj, obj.sx2layers['Emission'])
        vert_dict = self.vertex_data_dict(obj, masklayer, dots=False)

        if vert_dict and not empty:
            mod_vis = [modifier.show_viewport for modifier in obj.modifiers]
            for modifier in obj.modifiers:
                modifier.show_viewport = False

            hemi_up = Vector((0.0, 0.0, 1.0))
            vert_dict = self.vertex_data_dict(obj, masklayer, dots=False)
            face_colors = calculate_face_colors(obj)
            original_emissive_vertex_colors = {}
            original_emissive_vertex_face_count = [0 for _ in obj.data.vertices]
            dist = max(utils.get_object_bounding_box([obj, ], local=True)) * 5
            bias = 0.001

            for face in obj.data.polygons:
                color = face_colors[face.index]
                if color.length > 0:
                    for vert_idx in face.vertices:
                        vert_color = original_emissive_vertex_colors.get(vert_idx, Vector((0.0, 0.0, 0.0, 0.0)))
                        original_emissive_vertex_colors[vert_idx] = vert_color + color
                        original_emissive_vertex_face_count[vert_idx] += 1

            # Pass 1: Propagate emission to face colors
            hemisphere = self.ray_randomizer(raycount)
            contribution = 1.0 / float(raycount)
            for i in range(10):
                for j, face in enumerate(obj.data.polygons):
                    face_emission = Vector((0.0, 0.0, 0.0, 0.0))
                    vertices = [obj.data.vertices[face_vert_id].co for face_vert_id in face.vertices]
                    face_center = (sum(vertices, Vector()) / len(vertices)) + (bias * face.normal)
                    rotQuat = hemi_up.rotation_difference(face.normal)

                    for sample, _ in hemisphere:
                        sample_ray = rotQuat @ sample
                        hit, _, hit_normal, hit_face_index = obj.ray_cast(face_center, sample_ray, distance=dist)
                        if hit and (hit_normal.dot(sample_ray) < 0):
                            face_color = face_colors[hit_face_index].copy()
                            addition = face_color * contribution
                            face_emission += addition

                    if face_emission.length > face_colors[j].length:
                        face_colors[j] = face_emission.copy()

            # Pass 2: Average face colors to vertices
            vertex_colors = [Vector((0, 0, 0, 0)) for _ in obj.data.vertices]
            vertex_faces = [[] for _ in obj.data.vertices]
            vert_emission_list = self.empty_list(obj, 4)

            for face in obj.data.polygons:
                color = face_colors[face.index]
                if color.length > 0:
                    for vert_idx in face.vertices:
                        vertex_colors[vert_idx] += color
                        vertex_faces[vert_idx].append(face.index)

            for vert_idx, color_sum in enumerate(vertex_colors):
                if vert_idx in original_emissive_vertex_colors:
                    vertex_colors[vert_idx] = original_emissive_vertex_colors[vert_idx] / original_emissive_vertex_face_count[vert_idx]
                else:
                    if len(vertex_faces[vert_idx]) > 0:
                        vertex_colors[vert_idx] = color_sum / len(vertex_faces[vert_idx])

            for loop in obj.data.loops:
                vert_emission_list[(0+loop.index*4):(4+loop.index*4)] = vertex_colors[loop.vertex_index]

            # vert_emission_list = face_colors_to_loop_list(obj, face_colors)
            result = self.mask_list(obj, vert_emission_list, masklayer)

            for i, modifier in enumerate(obj.modifiers):
                modifier.show_viewport = mod_vis[i]

            return result
        else:
            return None


    def mask_list(self, obj, colors, masklayer=None, maskcolor=None, as_tuple=False, override_mask=False):
        count = len(colors)//4

        # No mask, colors pass through
        if (masklayer is None) and (maskcolor is None) and (sxglobals.mode != 'EDIT'):
            if as_tuple:
                rgba = [None] * count
                for i in range(count):
                    rgba[i] = tuple(colors[(0+i*4):(4+i*4)])
                return rgba
            else:
                return colors

        # Colors are masked by selection or layer alpha
        else:
            if masklayer is None:
                mask, empty = self.get_selection_mask(obj, selected_color=maskcolor)
                if empty:
                    return None
            # Layer is locked and there is an edit mode component selection
            elif (masklayer is not None) and (sxglobals.mode == 'EDIT'):
                mask1, empty = self.get_selection_mask(obj, selected_color=maskcolor)
                if empty:
                    return None
                else:
                    mask2, empty = layers.get_layer_mask(obj, masklayer)
                    if empty:
                        mask = mask1
                    else:
                        mask = [0.0] * len(mask1)
                        for i in range(len(mask1)):
                            if (mask1[i] > 0.0) and (mask2[i] > 0.0):
                                mask[i] = min(mask1[i], mask2[i]) 
            else:
                mask, empty = layers.get_layer_mask(obj, masklayer)
                if empty:
                    return None

            if as_tuple:
                rgba = [None] * count
                for i in range(count):
                    rgba[i] = tuple(Vector(colors[(0+i*4):(4+i*4)]) * mask[i])
                return rgba
            else:
                color_list = [None, None, None, None] * count
                if override_mask:
                    for i in range(count):
                        color = colors[(0+i*4):(4+i*4)]
                        color[3] = mask[i]
                        color_list[(0+i*4):(4+i*4)] = color
                else:
                    for i in range(count):
                        color = colors[(0+i*4):(4+i*4)]
                        color[3] *= mask[i]
                        color_list[(0+i*4):(4+i*4)] = color

                return color_list


    def color_list(self, obj, color, masklayer=None, as_tuple=False):
        count = len(obj.data.loops)
        colors = [color[0], color[1], color[2], color[3]] * count
        return self.mask_list(obj, colors, masklayer, as_tuple=as_tuple)


    def ramp_list(self, obj, objs, rampmode, masklayer=None, mergebbx=True):
        ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['Color Ramp']

        # For OBJECT mode selections
        if sxglobals.mode == 'OBJECT':
            if mergebbx:
                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box(objs)
            else:
                xmin, xmax, ymin, ymax, zmin, zmax = utils.get_object_bounding_box([obj, ])

        # For EDIT mode multi-obj component selection
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = utils.get_selection_bounding_box(objs)

        xdiv = float(xmax - xmin) or 1.0
        ydiv = float(ymax - ymin) or 1.0
        zdiv = float(zmax - zmin) or 1.0

        vertPosDict = self.vertex_data_dict(obj, masklayer)
        ramp_dict = {}

        for vert_id in vertPosDict:
            ratioRaw = None
            ratio = None
            fvPos = vertPosDict[vert_id][2]

            if rampmode == 'X':
                ratioRaw = ((fvPos[0] - xmin) / xdiv)
            elif rampmode == 'Y':
                ratioRaw = ((fvPos[1] - ymin) / ydiv)
            elif rampmode == 'Z':
                ratioRaw = ((fvPos[2] - zmin) / zdiv)

            ratio = max(min(ratioRaw, 1.0), 0.0)
            ramp_dict[vert_id] = ramp.color_ramp.evaluate(ratio)

        ramp_list = self.vert_dict_to_loop_list(obj, ramp_dict, 4, 4)

        return self.mask_list(obj, ramp_list, masklayer)


    def luminance_remap_list(self, obj, layer=None, masklayer=None, values=None):
        ramp = bpy.data.materials['SXToolMaterial'].node_tree.nodes['Color Ramp']

        if values is None:
            values = layers.get_luminances(obj, layer, as_rgba=False)
        colors = generate.empty_list(obj, 4)
        count = len(values)

        for i in range(count):
            ratio = max(min(values[i], 1.0), 0.0)
            colors[(0+i*4):(4+i*4)] = ramp.color_ramp.evaluate(ratio)

        return self.mask_list(obj, colors, masklayer)


    def vertex_id_list(self, obj):
        ids = [None] * len(obj.data.vertices)
        obj.data.vertices.foreach_get('index', ids)
        return ids


    def empty_list(self, obj, channelcount):
        return [0.0] * len(obj.data.loops) * channelcount


    def vert_dict_to_loop_list(self, obj, vert_dict, dictchannelcount, listchannelcount):
        mesh = obj.data
        loop_list = self.empty_list(obj, listchannelcount)

        if dictchannelcount < listchannelcount:
            if (dictchannelcount == 1) and (listchannelcount == 2):
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, 0.0)
                        loop_list[(0+loop_idx*listchannelcount):(listchannelcount+loop_idx*listchannelcount)] = [value, value]
            elif (dictchannelcount == 1) and (listchannelcount == 4):
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, 0.0)
                        loop_list[(0+loop_idx*listchannelcount):(listchannelcount+loop_idx*listchannelcount)] = [value, value, value, 1.0]
            elif (dictchannelcount == 3) and (listchannelcount == 4):
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, [0.0, 0.0, 0.0])
                        loop_list[(0+loop_idx*listchannelcount):(listchannelcount+loop_idx*listchannelcount)] = [value[0], value[1], value[2], 1.0]
        else:
            if listchannelcount == 1:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        loop_list[loop_idx] = vert_dict.get(vert_idx, 0.0)
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        loop_list[(0+loop_idx*listchannelcount):(listchannelcount+loop_idx*listchannelcount)] = vert_dict.get(vert_idx, [0.0] * listchannelcount)

        return loop_list


    def vertex_data_dict(self, obj, masklayer=None, dots=False):

        def add_to_dict(vert_id):
            min_dot = None
            if dots:
                dot_list = []
                vert = bm.verts[vert_id]
                num_connected = len(vert.link_edges)
                if num_connected > 0:
                    for edge in vert.link_edges:
                        dot_list.append((vert.normal.normalized()).dot((edge.other_vert(vert).co - vert.co).normalized()))
                min_dot = min(dot_list)

            vertex_dict[vert_id] = (
                mesh.vertices[vert_id].co,
                mesh.vertices[vert_id].normal,
                mat @ mesh.vertices[vert_id].co,
                (mat @ mesh.vertices[vert_id].normal - mat @ Vector()).normalized(),
                min_dot
            )


        mesh = obj.data
        mat = obj.matrix_world
        ids = self.vertex_id_list(obj)

        if dots:
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.normal_update()
            bmesh.types.BMVertSeq.ensure_lookup_table(bm.verts)

        vertex_dict = {}
        if masklayer is not None:
            mask, empty = layers.get_layer_mask(obj, masklayer)
            if not empty:
                for poly in mesh.polygons:
                    for vert_id, loop_idx in zip(poly.vertices, poly.loop_indices):
                        if mask[loop_idx] > 0.0:
                            add_to_dict(vert_id)

        elif sxglobals.mode == 'EDIT':
            vert_sel = [None] * len(mesh.vertices)
            mesh.vertices.foreach_get('select', vert_sel)
            if True in vert_sel:
                for vert_id, sel in enumerate(vert_sel):
                    if sel:
                        add_to_dict(vert_id)

        else:
            for vert_id in ids:
                add_to_dict(vert_id)

        if dots:
            bm.free()

        return vertex_dict


    # Returns mask and empty status
    def get_selection_mask(self, obj, selected_color=None):
        mesh = obj.data
        mask = self.empty_list(obj, 1)

        if selected_color is None:
            pre_check = [None] * len(mesh.vertices)
            mesh.vertices.foreach_get('select', pre_check)
            if True not in pre_check:
                empty = True
            else:
                empty = False
                if bpy.context.tool_settings.mesh_select_mode[2]:
                    for poly in mesh.polygons:
                        for loop_idx in poly.loop_indices:
                            mask[loop_idx] = float(poly.select)
                else:
                    for poly in mesh.polygons:
                        for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                            mask[loop_idx] = float(mesh.vertices[vert_idx].select)
        else:
            export.composite_color_layers([obj, ])
            colors = layers.get_layer(obj, obj.sx2layers['Composite'])

            if bpy.context.tool_settings.mesh_select_mode[2]:
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        mask[loop_idx] = float(utils.color_compare(colors[(0+loop_idx*4):(4+loop_idx*4)], selected_color, 0.01))
            else:
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        mask[loop_idx] = float(utils.color_compare(colors[(0+loop_idx*4):(4+loop_idx*4)], selected_color, 0.01))

            empty = 1.0 not in mask

        return mask, empty


    def __del__(self):
        print('SX Tools: Exiting generate')




# ------------------------------------------------------------------------
#    Layer Functions
#    NOTE: Objects must be in OBJECT mode before calling layer functions,
#          use utils.mode_manager() before calling layer functions
#          to set and track correct state
# ------------------------------------------------------------------------
class SXTOOLS2_layers(object):
    def __init__(self):
        return None


    def add_layer(self, objs, name=None, layer_type=None, color_attribute=None, clear=True):
        prefs = bpy.context.preferences.addons['sxtools2'].preferences
        data_types = {'FLOAT': 'FLOAT_COLOR', 'BYTE': 'BYTE_COLOR'}
        alpha_mats = {'OCC': 'Occlusion', 'MET': 'Metallic', 'RGH': 'Roughness', 'TRN': 'Transmission'}
        layer_dict = {}
        if objs:
            # Use largest layercount to avoid material mismatches
            for obj in objs:
                layercount = max([obj.sx2.layercount for obj in objs])

            for obj in objs:
                layer = obj.sx2layers.add()
                layer.name = 'Layer ' + str(layercount) if name is None else name
                if layer_type not in alpha_mats:
                    layer.color_attribute = color_attribute if color_attribute is not None else layer.name
                else:
                    layer.color_attribute = 'Alpha Materials'
                layer.layer_type = 'COLOR' if layer_type is None else layer_type
                layer.default_color = sxglobals.default_colors[layer.layer_type]
                # item.paletted is False by default

                if layer_type not in alpha_mats:
                    if color_attribute not in obj.data.color_attributes.keys():
                        obj.data.color_attributes.new(name=layer.name, type=data_types[prefs.layerdatatype], domain='CORNER')
                elif ('Alpha Materials' not in obj.data.color_attributes.keys()) and (layer_type in alpha_mats):
                    obj.data.color_attributes.new(name='Alpha Materials', type=data_types[prefs.layerdatatype], domain='CORNER')

                if layer_type == 'CMP':
                    layer.index = utils.insert_layer_at_index(obj, layer, 0)
                else:
                    layer.index = utils.insert_layer_at_index(obj, layer, obj.sx2layers[obj.sx2.selectedlayer].index + 1)

                if clear:
                    colors = generate.color_list(obj, layer.default_color)
                    layers.set_layer(obj, colors, obj.sx2layers[len(obj.sx2layers) - 1])
                layer_dict[obj] = layer
                obj.sx2.layercount = layercount + 1

            # selectedlayer needs to point to an existing layer on all objs
            objs[0].sx2.selectedlayer = len(objs[0].sx2layers) - 1

        return layer_dict


    def del_layer(self, objs, layer_name):
        for obj in objs:
            bottom_layer_index = obj.sx2layers[layer_name].index - 1 if obj.sx2layers[layer_name].index - 1 > 0 else 0
            if obj.sx2layers[layer_name].color_attribute == 'Alpha Materials':
                alpha_mat_count = sum(1 for layer in obj.sx2layers if layer.color_attribute == 'Alpha Materials')
                if alpha_mat_count == 1:
                    obj.data.attributes.remove(obj.data.attributes[obj.sx2layers[layer_name].color_attribute])
            else:
                if obj.sx2layers[layer_name].color_attribute in obj.data.color_attributes.keys():
                    obj.data.color_attributes.remove(obj.data.color_attributes[obj.sx2layers[layer_name].color_attribute])
            idx = utils.find_layer_index_by_name(obj, layer_name)
            obj.sx2layers.remove(idx)

        for obj in objs:
            utils.sort_stack_indices(obj)

        for i, layer in enumerate(objs[0].sx2layers):
            if layer.index == bottom_layer_index:
                objs[0].sx2.selectedlayer = i


    # wrapper for low-level functions, always returns layerdata in RGBA
    def get_layer(self, obj, sourcelayer, as_tuple=False, single_as_alpha=False, apply_layer_opacity=False):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        dv = [1.0, 1.0, 1.0, 1.0]

        if sourcelayer.layer_type in rgba_targets:
            values = self.get_colors(obj, sourcelayer.color_attribute)

        elif sourcelayer.layer_type in alpha_targets:
            source_values = self.get_colors(obj, sourcelayer.color_attribute)
            values = [None] * len(source_values)
            for i in range(len(source_values)//4):
                value_slice = source_values[(0+i*4):(4+i*4)]
                value = value_slice[alpha_targets[sourcelayer.layer_type]]
                if single_as_alpha:
                    if value > 0.0:
                        values[(0+i*4):(4+i*4)] = [dv[0], dv[1], dv[2], value]
                    else:
                        values[(0+i*4):(4+i*4)] = [0.0, 0.0, 0.0, value]
                else:
                    values[(0+i*4):(4+i*4)] = [value, value, value, 1.0]

        if apply_layer_opacity and sourcelayer.opacity != 1.0:
            count = len(values)//4
            for i in range(count):
                values[3+i*4] *= sourcelayer.opacity

        if as_tuple:
            count = len(values)//4
            rgba = [None] * count
            for i in range(count):
                rgba[i] = tuple(values[(0+i*4):(4+i*4)])
            return rgba

        else:
            return values


    # takes RGBA buffers, converts and writes to appropriate channels
    def set_layer(self, obj, colors, targetlayer):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        target_type = targetlayer.layer_type

        if target_type in rgba_targets:
            layers.set_colors(obj, targetlayer.color_attribute, colors)

        elif target_type in alpha_targets:
            target_values = self.get_colors(obj, 'Alpha Materials')
            values = layers.get_luminances(obj, sourcelayer=None, colors=colors, as_rgba=False)
            for i in range(len(values)):
                target_values[alpha_targets[target_type]+i*4] = values[i]
            self.set_colors(obj, 'Alpha Materials', target_values)


    def get_layer_mask(self, obj, sourcelayer):
        rgba_targets = ['COLOR', 'SSS', 'EMI', 'CMP']
        alpha_targets = {'OCC': 0, 'MET': 1, 'RGH': 2, 'TRN': 3}
        layer_type = sourcelayer.layer_type

        if layer_type in rgba_targets:
            colors = self.get_colors(obj, sourcelayer.color_attribute)
            values = colors[3::4]
        elif layer_type in alpha_targets:
            colors = self.get_colors(obj, sourcelayer.color_attribute)
            values = colors[alpha_targets[layer_type]::4]

        if any(v != 0.0 for v in values):
            return values, False
        else:
            return values, True


    def get_colors(self, obj, source_name):
        source_colors = obj.data.color_attributes[source_name].data
        colors = [None] * len(source_colors) * 4
        source_colors.foreach_get('color', colors)
        return colors


    def set_colors(self, obj, target, colors):
        target_colors = obj.data.color_attributes[target].data
        target_colors.foreach_set('color', colors)
        obj.data.update()


    def set_alphas(self, obj, target, values):
        colors = self.get_colors(obj, target)
        count = len(values)
        for i in range(count):
            color = colors[(0+i*4):(4+i*4)]
            color = [color[0], color[1], color[2], values[i]]
            colors[(0+i*4):(4+i*4)] = color
        target_colors = obj.data.color_attributes[target].data
        target_colors.foreach_set('color', colors)
        obj.data.update()  


    def get_luminances(self, obj, sourcelayer=None, colors=None, as_rgba=False, as_alpha=False):
        if colors is None:
            if sourcelayer is not None:
                colors = self.get_layer(obj, sourcelayer)
            else:
                colors = generate.empty_list(obj, 4)

        if as_rgba:
            values = generate.empty_list(obj, 4)
            count = len(values)//4
            for i in range(count):
                values[(0+i*4):(4+i*4)] = convert.luminance_to_color(convert.color_to_luminance(colors[(0+i*4):(4+i*4)]))
        elif as_alpha:
            values = generate.empty_list(obj, 4)
            count = len(values)//4
            for i in range(count):
                values[(0+i*4):(4+i*4)] = convert.luminance_to_alpha(convert.color_to_luminance(colors[(0+i*4):(4+i*4)]))
        else:
            values = generate.empty_list(obj, 1)
            count = len(values)
            for i in range(count):
                values[i] = convert.color_to_luminance(colors[(0+i*4):(4+i*4)])

        return values


    def get_uvs(self, obj, source, channel=None):
        channels = {'U': 0, 'V': 1}
        source_uvs = obj.data.uv_layers[source].data
        count = len(source_uvs)
        source_values = [None] * count * 2
        source_uvs.foreach_get('uv', source_values)

        if channel is None:
            uvs = source_values
        else:
            uvs = [None] * count
            sc = channels[channel]
            for i in range(count):
                uvs[i] = source_values[sc+i*2]

        return uvs


    # when targetchannel is None, sourceuvs is expected to contain data for both U and V
    def set_uvs(self, obj, target, sourceuvs, targetchannel=None):
        channels = {'U': 0, 'V': 1}
        target_uvs = obj.data.uv_layers[target].data

        if targetchannel is None:
            target_uvs.foreach_set('uv', sourceuvs)
        else:
            target_values = self.get_uvs(obj, target)
            tc = channels[targetchannel]
            count = len(sourceuvs)
            for i in range(count):
                target_values[tc+i*2] = sourceuvs[i]
            target_uvs.foreach_set('uv', target_values)


    def paste_layer(self, objs, targetlayer, fillmode):
        utils.mode_manager(objs, set_mode=True, mode_id='paste_layer')

        if fillmode == 'mask':
            for obj in objs:
                colors = layers.get_layer(obj, targetlayer)
                alphas = sxglobals.copy_buffer[obj.name]
                count = len(colors)//4
                for i in range(count):
                    colors[(3+i*4):(4+i*4)] = alphas[(3+i*4):(4+i*4)]
                layers.set_layer(obj, colors, targetlayer)
        elif fillmode == 'lumtomask':
            for obj in objs:
                colors = layers.get_layer(obj, targetlayer)
                alphas = convert.colors_to_values(sxglobals.copy_buffer[obj.name])
                count = len(colors)//4
                for i in range(count):
                    colors[3+i*4] = alphas[i]
                layers.set_layer(obj, colors, targetlayer)
        else:
            for obj in objs:
                colors = sxglobals.copy_buffer[obj.name]
                targetvalues = self.get_layer(obj, targetlayer)
                if sxglobals.mode == 'EDIT':
                    colors = generate.mask_list(obj, colors)
                colors = tools.blend_values(colors, targetvalues, 'ALPHA', 1.0)
                layers.set_layer(obj, colors, targetlayer)

        utils.mode_manager(objs, set_mode=False, mode_id='paste_layer')


    def __del__(self):
        print('SX Tools: Exiting layers')



# ------------------------------------------------------------------------
#    UI Elements
# ------------------------------------------------------------------------
class SXCHANNELER_LayerItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    type: bpy.props.StringProperty()  # "UV" or "COLOR"


class SXCHANNELER_UL_LayersList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=f"{item.type}: {item.name}")


class SXCHANNELER_PT_main_panel(bpy.types.Panel):
    bl_label = "Copy Paste Channels"
    bl_idname = "SXCHANNELER_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Copy Paste Channels'


    @classmethod
    def poll(cls, context):
        obj = context.view_layer.objects.active
        return obj is not None


    def draw(self, context):
        layout = self.layout
        obj = context.view_layer.objects.active
        if obj.type == 'MESH':
            layout.template_list("SXCHANNELER_UL_LayersList", "", obj, "sxchanneler_layers", obj, "sxchanneler_layer_index")

            row = layout.row()
            row.operator('channel.copy')

            row = layout.row()
            row.operator('channel.paste')
        else:
            layout.label(text='No mesh object selected')


class SXCHANNELER_OT_copy(bpy.types.Operator):
    bl_idname = "channel.copy"
    bl_label = "Copy Channels"

    def execute(self, context):
        obj = context.object
        if obj.type == 'MESH':
            global stored_data
            item = obj.LayerList[obj.layer_index]
            if item.type == 'COLOR':
                stored_data['colors'][item.name] = obj.data.vertex_colors[item.name].data[:]
            elif item.type == 'UV':
                stored_data['uvs'][item.name] = obj.data.uv_layers[item.name].data[:]
        return {'FINISHED'}


class SXCHANNELER_OT_paste(bpy.types.Operator):
    bl_idname = "channel.paste"
    bl_label = "Paste Channels"

    def execute(self, context):
        obj = context.object
        if obj.type == 'MESH':
            global stored_data
            item = obj.LayerList[obj.layer_index]
            if item.type == 'COLOR' and item.name in stored_data['colors']:
                obj.data.vertex_colors[item.name].data[:] = stored_data['colors'][item.name]
            elif item.type == 'UV' and item.name in stored_data['uvs']:
                obj.data.uv_layers[item.name].data[:] = stored_data['uvs'][item.name]
        return {'FINISHED'}


def update_layers_list(obj, context):
    if obj.type == 'MESH':
        obj.sxchanneler_layers.clear()
        for layer in obj.data.vertex_colors:
            item = obj.sxchanneler_layers.add()
            item.name = layer.name
            item.type = 'COLOR'
        for layer in obj.data.uv_layers:
            item = obj.sxchanneler_layers.add()
            item.name = layer.name
            item.type = 'UV'


def register():
    bpy.utils.register_class(SXCHANNELER_LayerItem)
    bpy.utils.register_class(SXCHANNELER_UL_LayersList)
    bpy.utils.register_class(SXCHANNELER_OT_copy)
    bpy.utils.register_class(SXCHANNELER_OT_paste)
    bpy.utils.register_class(SXCHANNELER_PT_main_panel)

    bpy.types.Object.sxchanneler_layers = bpy.props.CollectionProperty(type=SXCHANNELER_LayerItem)
    bpy.types.Object.sxchanneler_layer_index = bpy.props.IntProperty()
    bpy.types.Scene.update_active = bpy.props.BoolProperty(update=update_layers_list)


def unregister():
    bpy.utils.unregister_class(SXCHANNELER_LayerItem)
    bpy.utils.unregister_class(SXCHANNELER_UL_LayersList)
    bpy.utils.unregister_class(SXCHANNELER_OT_copy)
    bpy.utils.unregister_class(SXCHANNELER_OT_paste)
    bpy.utils.unregister_class(SXCHANNELER_PT_main_panel)

    del bpy.types.Object.sxchanneler_layers
    del bpy.types.Object.sxchanneler_layer_index
    del bpy.types.Scene.update_active


if __name__ == "__main__":
    register()
