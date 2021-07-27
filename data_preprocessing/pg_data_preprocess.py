import os
import cv2
import json
import bezier
import numpy as np

def four00Points(points):
    new_points = []
    for i in range(len(points) - 1):
        pi, pj = i, i + 1
        new_points.append(points[pi])
        center = ((points[pi][0] + points[pj][0]) / 2, (points[pi][1] + points[pj][1]) / 2)
        new_points.append(center)
    new_points.append(points[-1])
    return new_points

def approximate_curved_polygon(_contour, point_num=200):
    """
    使用贝塞尔曲线进行拟合,得到平滑的闭合多边形轮廓
    :param _contour: 构成多边形轮廓的点集. Array:(N, 2)
    :param point_num: 每次拟合的点的数量,越大则越平滑. Int
    :return: 返回平滑后的轮廓点
    """
    to_return_contour = []
    _contour = np.reshape(_contour, (-1, 2))
    # 复制起始点到最后,保证生成闭合的曲线
    # _contour = np.vstack((_contour, _contour[0, :].reshape((-1, 2))))
    for start_index in range(0, _contour.shape[0], point_num):
        # 多取一个点,防止曲线中间出现断点
        end_index = start_index + point_num + 1
        end_index = end_index if end_index < _contour.shape[0] else _contour.shape[0]
        nodes = np.transpose(_contour[start_index:end_index, :])
        # 拟合贝塞尔曲线
        curve = bezier.Curve(nodes, degree=nodes.shape[1] - 1)
        curve = curve.evaluate_multi(np.linspace(0.0, 1.0, point_num * 5))
        to_return_contour.append(np.transpose(curve))
    to_return_contour = np.array(to_return_contour).reshape((-1, 2))
    return to_return_contour

def select_seven_points(ori_points):
    points = ori_points
    while len(points) < 40:
        points = four00Points(points)
    points = np.array(points)
    res_point = approximate_curved_polygon(points)
    num = (len(res_point)) // 6
    mod = len(res_point) % 6
    new_points = []
    i = 0
    count = 0
    while i < len(res_point):
        new_points.append(res_point[i])
        if count == mod:
            i += num
        else:
            i += num + 1
            count += 1
    new_points.append(ori_points[-1])
    new_points = np.array(new_points)
    return new_points

def feed_data(image_dict, data):
    image_name, number, content, new_points, top_line, end_line = data
    if number in image_dict[image_name]:
        if top_line:
            image_dict[image_name][number].update({"content": content, "top_line": new_points})
        else:
            image_dict[image_name][number].update({"end_line": new_points})
    else:
        if top_line:
            image_dict[image_name].update({number:{"content": content, "top_line": new_points}})
        else:
            image_dict[image_name].update({number:{"end_line": new_points}})
    return image_dict

def write_label_with_PgNet(fp, label_dict):
    write_line = ""
    for image_name in label_dict.keys():
        try:
            write_line = image_name + "\t"
            label_str = ""
            for num in label_dict[image_name].keys():
                content = label_dict[image_name][num]["content"]
                top_line = label_dict[image_name][num]["top_line"].tolist()
                end_line = label_dict[image_name][num]["end_line"].tolist()
                points = top_line + end_line
                write_str = "{" + '"{}": "{}", "{}": {}'.format("transcription", content, "points", points) + "}"
                if len(label_str) < 1:
                    label_str = write_str
                else:
                    label_str = label_str + ',' + write_str
            write_line = write_line + "[" + label_str + "]" + "\n"
        except Exception as e:
            print(e)
    fp.write(write_line)

def data_preproces(root_dir):
    '''
    root_dir: 目录下存放标注的文件
    '''
    fw = open("{}_label.txt".format(root_dir), "w")
    for sub_dir_root in os.listdir(root_dir):
        print(sub_dir_root)
        image_dir = os.path.join(root_dir, sub_dir_root)
        if not os.path.isdir(image_dir):
            continue
        json_file = '{}.json'.format(image_dir)
        with open(json_file) as fr:
            label_dict = json.load(fr)
        img_metadata = label_dict['_via_img_metadata']
        for key in img_metadata.keys():
            image_dict = {}
            sub_dir = img_metadata[key]
            image_name = sub_dir['filename']
            image_dict[image_name] = {}
            image = cv2.imread(os.path.join(image_dir, image_name))
            if image is None:
                continue
            regions = sub_dir["regions"]
            if len(regions) < 1:
                continue
            for sub_regions in regions:
                try:
                    shape_attributes = sub_regions["shape_attributes"]
                    all_points_x = shape_attributes["all_points_x"]
                    all_points_y = shape_attributes["all_points_y"]
                    points = np.vstack((all_points_x, all_points_y))
                    points = np.transpose(points)
                    cv2.polylines(image, [points], False, (255, 0, 0), 5, 1)
                    new_points = select_seven_points(points)        # 平均选7个点
                    region_attributes = sub_regions["region_attributes"]
                    number = region_attributes["number"]
                    top_line, end_line = False, False
                    if "top_line" in region_attributes["line"].keys():
                        content = region_attributes["content"]      # 选取文本信息
                        top_line = True
                    if "end_line" in region_attributes["line"].keys():
                        end_line = True
                        content = ""
                    if "\n" in content:
                        print(image_name)
                        print("###################")
                        exit()
                    image_dict  = feed_data(image_dict, [image_name, number, content, new_points, top_line, end_line])
                except Exception as e:
                    print(image_name)
                    print(e, "@@@@@@@@@@@@@@@@@@@@@@@@")
            write_label_with_PgNet(fw, image_dict)
    fw.close()

def main():
    data_preproces()

if __name__ == '__main__':
    main()