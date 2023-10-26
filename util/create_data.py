from pymongo import MongoClient
import pandas

class OlpDataGenerator():
    def __init__(self, client: MongoClient, db_name: str, collection_name: str) -> None:
        self.db = client[db_name]
        self.ik_result = self.db[collection_name]
        self._refresh()
        # self.docs = self.ik_result.find({})

    def generate_graph(self):
        self._refresh()
        root_name = 'factory'
        self.df = pandas.DataFrame(columns=['id1', 'id2', 'weight'])
        goal_count_df = pandas.DataFrame(columns=['id', 'count'])

        for doc in self.docs:
            station_name = self._get_station_name(doc)
            robot_name = self._get_robot_name(doc)
            tool_name = self._get_tool_name(doc)
            goal_name = self._get_goal_name(doc)

            name_list = [goal_name, tool_name, robot_name, station_name, root_name]

            r = 2
            temp_store = ['']*r
            name_combination = []
            self._get_combination(name_list, temp_store, 0, len(name_list) - 1, 0, r, name_combination)

            for i in range(len(name_combination)):
                self._add_edge(id1=name_combination[i][0], id2=name_combination[i][1])

    def export_graph(self, filename: str) -> None:
        self.df.to_csv(filename, index=False)

    def count_id(self, category) -> pandas.DataFrame:
        self._refresh()
        idcount_df = pandas.DataFrame(columns=['id', 'count'])

        for doc in self.docs:
            if category == 'goal':
                id_name = self._get_goal_name(doc)
            elif category == 'tool':
                id_name = self._get_tool_name(doc)
            elif category == 'robot':
                id_name = self._get_robot_name(doc)
            elif category == 'station':
                id_name = self._get_station_name(doc)
            else:
                ValueError('wrong id name')

            id_exp = "(id == @id_name)"
            if idcount_df.query(id_exp).empty:
                new_row = pandas.Series({'id': id_name, 'count': 1})
                idcount_df = pandas.concat([idcount_df, new_row.to_frame().T], ignore_index=True)
            else:
                for q_ind in idcount_df.query(id_exp).index.values:
                    idcount_df.iloc[q_ind, :]['count'] = idcount_df.iloc[q_ind, :]['count'] + 1

        return idcount_df
    
    def get_tool_list_for_goals(self) -> pandas.DataFrame:
        self._refresh()
        goal_tool_list_df = pandas.DataFrame(columns=['id', 'count', 'tool_list'])

        for doc in self.docs:
            id_name = self._get_goal_name(doc)
            id_exp = "(id == @id_name)"
            if goal_tool_list_df.query(id_exp).empty:
                new_row = pandas.Series({'id': id_name, 'count': 1, 'tool_list': [self._get_tool_name(doc)]})
                goal_tool_list_df = pandas.concat([goal_tool_list_df, new_row.to_frame().T], ignore_index=True)
            else:
                for q_ind in goal_tool_list_df.query(id_exp).index.values:
                    tool_list = goal_tool_list_df.iloc[q_ind, :]['tool_list']
                    tool_name = self._get_tool_name(doc)
                    tool_list.append(tool_name)
                    goal_tool_list_df.iloc[q_ind, :]['tool_list'] = tool_list
                    goal_tool_list_df.iloc[q_ind, :]['count'] = goal_tool_list_df.iloc[q_ind, :]['count'] + 1

        return goal_tool_list_df

    def get_hierarchy_names(self) -> pandas.DataFrame:
        self._refresh()
        root_name = 'factory'

        self.hierarchy_names = pandas.DataFrame(columns=['id'])
        new_row = pandas.Series({'id': root_name})
        self.hierarchy_names = pandas.concat([self.hierarchy_names, new_row.to_frame().T], ignore_index=True)

        for doc in self.docs:
            station_name = self._get_station_name(doc)
            robot_name = self._get_robot_name(doc)
            tool_name = self._get_tool_name(doc)
            goal_name = self._get_goal_name(doc)

            name_list = [goal_name, tool_name, robot_name, station_name, root_name]

            r = 2
            temp_store = ['']*r
            name_combination = []
            self._get_combination(name_list, temp_store, 0, len(name_list) - 1, 0, r, name_combination)

            for i in range(2, len(name_list) - 1):
                ig_id = name_list[i]
                str_exp = "(id == @ig_id)"
                if self.hierarchy_names.query(str_exp).empty:
                    new_row = pandas.Series({'id': ig_id})
                    self.hierarchy_names = pandas.concat([self.hierarchy_names, new_row.to_frame().T], ignore_index=True)

        return self.hierarchy_names


    def _refresh(self) -> None:
        self.docs = self.ik_result.find({})

    def _get_combination(self, arr, data, start, end, index, r, result) -> None:
        # ref: https://www.geeksforgeeks.org/print-all-possible-combinations-of-r-elements-in-a-given-array-of-size-n/
        """get all possible r number of combinations from arr.

        Args:
            arr (_type_): _description_
            data (_type_): _description_
            start (_type_): _description_
            end (_type_): _description_
            index (_type_): _description_
            r (_type_): _description_
            result (_type_): _description_
        """
        if index == r:
            temp_comb = []
            for j in range(r):
                # print(data[j], end=" ")
                temp_comb.append(data[j])
            # print()
            result.append(temp_comb)
            return
    
        i = start
        while (i <= end and end - i + 1 >= r - index):
            data[index] = arr[i]
            self._get_combination(arr, data, i + 1, end, index + 1, r, result)
            i += 1

    def _add_edge(self, id1, id2):
        str_exp = "(id1 == @id1) and (id2 == @id2)"
        if self.df.query(str_exp).empty:
            new_row = pandas.Series({'id1': id1, 'id2': id2, 'weight': 1})
            self.df = pandas.concat([self.df, new_row.to_frame().T], ignore_index=True)
            # self.df = self.df._append({'id1': id1, 'id2': id2, 'weight': 1}, ignore_index=True)

    def _get_station_name(self, doc) -> str:
        return doc['station_name']
    
    def _get_robot_name(self, doc) -> str:
        return 'R' + str(doc['robot_id']) + '.' + self._get_station_name(doc)
    
    def _get_tool_name(self, doc) -> str:
        return doc['tool_name'] + '.' + self._get_robot_name(doc)

    def _get_goal_name(self, doc) -> str:
        return doc['name']

    