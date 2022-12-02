import cv2
from shapely.geometry import Polygon

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.area
    return iou

def check_overlap(boxes_original, scores_original, classid_original):
    boxes = []
    scores = []
    classid = []
    for i, box in enumerate(boxes_original):
        append_true = True
        if classid_original[i] == 0:
            xmin, ymin, xmax, ymax = box
            box_license_point = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            for j, box_1 in enumerate(boxes_original):
                if classid_original[j] == 0 and i !=j:
                    xmin1, ymin1, xmax1, ymax1 = box_1
                    box_license_point_1 = [[xmin1, ymin1], [xmax1, ymin1], [xmax1, ymax1], [xmin1, ymax1]]
                    iou = calculate_iou(box_license_point, box_license_point_1)
                    if int(round(iou * 100, 2)) >= 95:
                        append_true = False
                        break
            if append_true == True:
                boxes.append(box)
                scores.append(scores_original[i])
                classid.append(classid_original[i])
    return boxes, scores, classid

def check_polygon_sort(polygon_license, polygon_license_checks, polygon_check):
    # polygon_check: vung detect
    # polygon_license_checks: tap hop cac box dang nam trong vung detect
    # polygon_license: box dang check
    if polygon_license_checks == []:
        polygon_license_checks.append(polygon_license)
        push = True
        # print("-------1-----------")
    else:
        # Xoa box cu khi da di ra khoi vung detect
        # for i, polygon_license_check in enumerate(polygon_license_checks):
        #     # Remove polygon
        #     intersect_check = polygon_license_check.intersection(polygon_check).area / polygon_license_check.area
        #     if int(round(intersect_check * 100, 2)) <= 80: 
        #         polygon_license_checks.remove(polygon_license_check)
        # Trong vung detect
        push = True
        for j, polygon_license_check in enumerate(polygon_license_checks):
            area = polygon_license.intersection(polygon_license_check).area / polygon_license.area
            # print("--------------------area---------------", int(round(area * 100, 2)))
            if int(round(area * 100, 2)) >= 25: #Cung 1 nguoi
                polygon_license_checks[j] = polygon_license
                push = False
                # print("-------2-----------")
                break
        if push == True:
            polygon_license_checks.append(polygon_license)
            # print("-------3-----------")
    return polygon_license_checks, push

def clear_polygons(boxes, polygon_check):
    clear_polygon = True
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        box_license_point = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        polygon_license = Polygon(box_license_point)
        intersect = polygon_license.intersection(polygon_check).area / polygon_license.area
        if int(round(intersect * 100, 2)) >= 90:
            clear_polygon = False
            break
    return clear_polygon