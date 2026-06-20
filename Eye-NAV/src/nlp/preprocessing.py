"""
This class is to preprocess, sort and clean up the data
to send to both LLM and default_responses. It is designed to 
give more specific instructions to the user, not just generic.

We can sort the objects themselves by various methods.

we have implemented logic for the following here:

1. by count
2. by distance (closest only if multiple)
3. by position in frame

4. BY USING ALL THE METRICS ABOVE IN ONE UNIFIED FUNCTION

"""
from typing import Dict, List
from collections import Counter, defaultdict



class Preprocessing:
    def __init__(self):
        # Initialize any variables here
        self.count_sorted_objects:Dict[str,int] = dict()
        self.distance_sorted_objects:Dict[str,int] =dict()
        self.all_sorted_objects:Dict[str,List[dict]] = dict()


    def count_objects(self, objects:list) ->dict:

        labels= [obj['label'] for obj in objects]
        counts= Counter(labels)

        #returns descending order (by count) of object:count mapping
        #e.g. {'person': 3, 'chair': 2, 'door': 1}
        return dict(counts.most_common())


    def sort_by_distance(self, objects:list) ->dict:

        #if there are multiple count of the same object type, 
        # we map it to the closest one 
        min_distances = {}
        for obj in objects:
            label = obj['label']
            dist_val = obj.get('distance', float('inf'))
            
            #update closest distance if label already existing
            # meaning more than one count of same type object

            if label not in min_distances or dist_val < min_distances[label]:
                min_distances[label] = dist_val

        #sort by distance
        sorted_by_distance = dict(sorted(min_distances.items(), key=lambda item: item[1]))
        
        #return dict of {'person' : 2.5, 'chair' : 1.2, ...}
        return sorted_by_distance


    def get_position_score(self, obj:dict)->int:

        # obviously this function assumes the camera is held upright
        # in a typical way of testing
        # assumptions made to simplify the way our pipeline works
        positions = obj.get("position")

        best_score = 3

        for p in positions:
            if "middle-center" in p or "low-center" in p:
                best_score = min(best_score, 1)
            elif "middle-left" in p or "middle-right" in p or "low-left" in p or "low-right" in p:
                best_score = min(best_score, 2)
            else:
                best_score = min(best_score, 3)        # for top-left, top-right, top-center

        return best_score

    def get_averaged_position_text(self, obj: dict) -> str:
        """
        Calculates the average geometric centroid spanning multiple bounding boxes.
        
        Why v_map and h_map are useful:
        They convert raw English position strings ("top", "middle", "left") into a 
        2D mathematical coordinate plane (-1, 0, 1). This allows us to calculate an 
        exact center of mass for a large object instead of just guessing from words.
        
        Example: 
        If an object has positions ['middle-left', 'middle-center', 'middle-right']:
        1. Vertical maps to [0, 0, 0]. Average = 0 -> "middle"
        2. Horizontal maps to [-1, 0, 1]. All 3 zones hit, so it spans the whole width.
        Final Output: "middle"  (dropping the horizontal because it spans everything)
        this is useful for when one person/object occupies majority of the fram
        """
        # function specifically for default_responses, nothing else

        positions = obj.get("position")
        if not positions:
            return "ahead"          #generic response that default_response was giving
            
        v_map = {'top': -1, 'middle': 0, 'low': 1}
        h_map = {'left': -1, 'center': 0, 'right': 1}
        
        v_rev = {-1: 'top', 0: 'middle', 1: 'low'}
        h_rev = {-1: 'left', 0: 'center', 1: 'right'}
        
        v_sum = 0
        h_sum = 0
        valid_positions = 0
        
        v_set = set()
        h_set = set()
        
        for p in positions:
            parts = p.split('-')
            if len(parts) == 2:     # it always will be by our current implementation
                v, h = parts[0], parts[1]
                v_sum += v_map.get(v, 0)
                h_sum += h_map.get(h, 0)
                v_set.add(v)
                h_set.add(h)
                valid_positions += 1
                
        if valid_positions == 0:  # generic response
            return " ".join(positions).replace('-', ' ') if positions else "ahead"
            

        # need to average out
        v_avg = round(v_sum / valid_positions)
        h_avg = round(h_sum / valid_positions)
        
        # 3. Final Output Assembly
        # We omit the vertical row labels (top, middle, low) to simplify instructions 
        # as requested, focusing only on the horizontal direction.
        
        # If the object spans the whole horizontal frame, it's just "ahead"
        if len(h_set) == 3:
            res = "ahead"
        else:
            h_text = h_rev.get(h_avg, "center")
            if h_text == "center":
                res = "ahead"
            else:
                res = h_text
            
        return res

    def _parse_distance(self, obj: dict) -> float:
        # returns distance
        return float(obj.get('distance', float('inf')))

    def dist_pos_sorting(self, objects:list)->dict:
        #return a dict as 
        """ "person": [closest_person_dict, 2nd_closest_person_dict, ...],
            "chair": [closest_chair_dict, ...]
        }
        """
        # 1. Use your existing count function to get keys natively in descending order
        counts_dict = self.count_objects(objects)
        
        # 2. Group the objects into buckets 
        # we have then, grouped = {'person': [obj1, obj2, obj3], 'chair': [obj4, obj5], ...}
        grouped = defaultdict(list)
        for obj in objects:
            grouped[obj['label']].append(obj)
            
        final_dict = {}
        
        # 3. Iterate over the already-sorted keys from count_objects
        for label in counts_dict.keys():
            # Get the list of objects for this label with all their attached labels
            obj_list = grouped[label]
            
            # Sort the objects inside it by distance
            # e.g. if obj_list = [obj1, obj2, obj3] 
            # obj1 is closest, obj2 is 2nd closest, etc.
           
            sorted_obj_list = sorted(obj_list, key=self._parse_distance)
            
            # Add to final dict
            #final_dict = {'person': [obj1, obj2, obj3], 'chair': [obj4, obj5], ...}
            final_dict[label] = sorted_obj_list
            
        return final_dict




    def execute(self, fused_state:dict, upcoming_turn_detected: bool):
        #recieves list of current objects 
        """
        structure of fused_state:
         fused_state = {
                    "timestamp": current_ts,
                    "count": len(fused_objects),
                    "objects": fused_objects,
                    "navigation_instruction": self.nav_state['path_instruction']
                }
        """
        
        #ALL THE VARIABLES WE RECIEVE AND USE FOR FURTHER ANALYSIS
        total_objects_count= fused_state['count']
        objects_list=fused_state['objects']
        navigation_instruction=fused_state['navigation_instruction']
        upcoming_turn_detected=upcoming_turn_detected
        
        # SORT BY ALL METRICS
        self.all_sorted_objects = self.dist_pos_sorting(objects_list)
        return self.all_sorted_objects
    

   
       
    
        