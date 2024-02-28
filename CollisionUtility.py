import pygame

'''
    Class created using code from https://www.pygame.org/wiki/IntersectingLineDetection
    Mathematic explanation: https://www.mathopenref.com/coordintersection.html
'''
class CollisionUtility:

    @staticmethod
    def check_lander_collision_with_surface(lander, surface):
        lander_bottom_line = [lander.rect.bottomleft, lander.rect.bottomright]
        lander_top_line = [lander.rect.topleft, lander.rect.topright]
        lander_left_line = [lander.rect.topleft, lander.rect.bottomleft]
        lander_right_line = [lander.rect.topright, lander.rect.bottomright]
        surface_points = CollisionUtility.surface_points_below_lander(lander, surface)

        intersection_point_found = False

        for i in range(len(surface_points)-1):
            top_intersect_point = CollisionUtility.calculateIntersectPoint(lander_top_line[0], lander_top_line[1], surface_points[i], surface_points[i+1])
            bottom_intersect_point = CollisionUtility.calculateIntersectPoint(lander_bottom_line[0], lander_bottom_line[1], surface_points[i], surface_points[i+1])
            left_intersect_point = CollisionUtility.calculateIntersectPoint(lander_left_line[0], lander_left_line[1], surface_points[i], surface_points[i+1])
            right_intersect_point = CollisionUtility.calculateIntersectPoint(lander_right_line[0], lander_right_line[1], surface_points[i], surface_points[i+1])
            if (bottom_intersect_point != None or top_intersect_point != None or left_intersect_point != None or right_intersect_point != None):
                intersection_point_found = True

        if (not intersection_point_found):
            lowest_lander_point = max(lander_bottom_line[0][1], lander_bottom_line[1][1], lander_top_line[0][1], lander_top_line[1][1])
            lowest_surface_point = 0
            for p in surface_points:
                lowest_surface_point = max(lowest_surface_point, p[1])
            intersection_point_found = (lowest_surface_point < lowest_lander_point)

        return intersection_point_found

    # Calc the gradient 'm' of a line between p1 and p2
    @staticmethod
    def calculateGradient(p1, p2):
    
        # Ensure that the line is not vertical
        if (p1[0] != p2[0]):
            m = (p1[1] - p2[1]) / (p1[0] - p2[0])
            return m
        else:
            return None

    # Calc the point 'b' where line crosses the Y axis
    @staticmethod
    def calculateYAxisIntersect(p, m):
        return  p[1] - (m * p[0])

    # Calc the point where two infinitely long lines (p1 to p2 and p3 to p4) intersect.
    # Handle parallel lines and vertical lines (the later has infinate 'm').
    # Returns a point tuple of points like this ((x,y),...)  or None
    # In non parallel cases the tuple will contain just one point.
    # For parallel lines that lay on top of one another the tuple will contain
    # all four points of the two lines
    @staticmethod
    def getIntersectPoint(p1, p2, p3, p4):
        m1 = CollisionUtility.calculateGradient(p1, p2)
        m2 = CollisionUtility.calculateGradient(p3, p4)
            
        # See if the the lines are parallel
        if (m1 != m2):
            # Not parallel
            
            # See if either line is vertical
            if (m1 is not None and m2 is not None):
                # Neither line vertical           
                b1 = CollisionUtility.calculateYAxisIntersect(p1, m1)
                b2 = CollisionUtility.calculateYAxisIntersect(p3, m2)   
                x = (b2 - b1) / (m1 - m2)       
                y = (m1 * x) + b1           
            else:
                    # Line 1 is vertical so use line 2's values
                if (m1 is None):
                    b2 = CollisionUtility.calculateYAxisIntersect(p3, m2)   
                    x = p1[0]
                    y = (m2 * x) + b2
                # Line 2 is vertical so use line 1's values               
                elif (m2 is None):
                    b1 = CollisionUtility.calculateYAxisIntersect(p1, m1)
                    x = p3[0]
                    y = (m1 * x) + b1           
                else:
                    assert False
                    
            return ((x,y),)
        else:
            # Parallel lines with same 'b' value must be the same line so they intersect
            # everywhere in this case we return the start and end points of both lines
            # the calculateIntersectPoint method will sort out which of these points
            # lays on both line segments
            b1, b2 = None, None # vertical lines have no b value
            if m1 is not None:
                b1 = CollisionUtility.calculateYAxisIntersect(p1, m1)
                
            if m2 is not None:   
                b2 = CollisionUtility.calculateYAxisIntersect(p3, m2)
            
            # If these parallel lines lay on one another   
            if b1 == b2:
                return p1,p2,p3,p4
            else:
                return None

    # For line segments (ie not infinitely long lines) the intersect point
    # may not lay on both lines.
    #   
    # If the point where two lines intersect is inside both line's bounding
    # rectangles then the lines intersect. Returns intersect point if the line
    # intesect o None if not
    @staticmethod
    def calculateIntersectPoint(p1, p2, p3, p4):
    
        p = CollisionUtility.getIntersectPoint(p1, p2, p3, p4)
    
        if p is not None:               
            width = p2[0] - p1[0]
            height = p2[1] - p1[1]       
            r1 = pygame.Rect(p1, (width , height))
            r1.normalize()
            
            width = p4[0] - p3[0]
            height = p4[1] - p3[1]
            r2 = pygame.Rect(p3, (width, height))
            r2.normalize()              

            # Ensure both rects have a width and height of at least 'tolerance' else the
            # collidepoint check of the Rect class will fail as it doesn't include the bottom
            # and right hand side 'pixels' of the rectangle
            tolerance = 1
            if r1.width < tolerance:
                r1.width = tolerance
                    
            if r1.height < tolerance:
                r1.height = tolerance
            
            if r2.width < tolerance:
                r2.width = tolerance
                        
            if r2.height < tolerance:
                r2.height = tolerance

            for point in p:                 
                try:    
                    res1 = r1.collidepoint(point)
                    res2 = r2.collidepoint(point)
                    if res1 and res2:
                        point = [int(pp) for pp in point]                                
                        return point
                except:
                    # sometimes the value in a point are too large for PyGame's Rect class
                    str = "point was invalid  ", point
                    print(str)
                    
            # This is the case where the infinately long lines crossed but 
            # the line segments didn't
            return None            
        else:
            return None
    
    @staticmethod
    def surface_points_below_lander(lander, surface):
        lander_leftmost_point = lander.rect.bottomleft[0]
        lander_rightmost_point = lander.rect.bottomright[0]
        points_below_lander = []
        leftmost_point_found = False
        rightmost_point_found = False 
        for i in range(len(surface.polygon_points)-1):
            if (not leftmost_point_found):
                p = surface.polygon_points[i]
                p1 = surface.polygon_points[i+1]
                if (p[0] <= lander_leftmost_point and p1[0] > lander_leftmost_point):
                    points_below_lander.append(p)
                    leftmost_point_found = True
            elif (not rightmost_point_found):
                p = surface.polygon_points[i]
                if (p[0] < lander_rightmost_point):
                    points_below_lander.append(p)
                else:
                    points_below_lander.append(p)
                    rightmost_point_found = True
        
        return points_below_lander

    @staticmethod
    def check_gameobject_window_collision(gameobject, screen_dimensions):
        gameobject_leftmost_point = gameobject.rect.topleft[0]
        gameobject_rightmost_point = gameobject.rect.topright[0]
        gameobject_bottommost_point = gameobject.rect.bottomleft[1]
        # Check left side of the window
        if (gameobject_rightmost_point < 0):
            return True
        # Check right side of the window
        elif (gameobject_leftmost_point > screen_dimensions[0]):
            return True
        # Check top side of the window 
        # there is no need to check bottom side since there will be collision with surface
        elif (gameobject_bottommost_point < 0):
            return True
        else:
            return False