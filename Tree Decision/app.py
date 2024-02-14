import streamlit as st
import pandas as pd
import math
import pydotplus
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from graphviz import Digraph


# import graphviz
# import streamlit_graphviz as sg
# from dtreeviz.trees import dtreeviz

def entropy(probs):
    return sum([-prob * math.log(prob, 2) for prob in probs])


def entropy_of_list(ls, value):

    total_instances = len(ls)
    st.write("---------------------------------------------------------")
    st.write(
        "\nعدد السجلات الكلي المرتبطة بـ '{0}' هو : {1}".format(value, total_instances)
    )

    cnt = Counter(x for x in ls)
    st.write("\nعدد فئات السمة المستهدفة (نعم / لا) =", dict(cnt))

    probs = [x / total_instances for x in cnt.values()]
    st.write("\nالفئات :", max(cnt), min(cnt))
    st.write("\nاحتمالية الفئة 'نعم' ='{0}' : {1}".format(max(cnt), max(probs)))
    st.write("احتمالية الفئة 'لا' ='{0}' : {1}".format(min(cnt), min(probs)))

    return entropy(probs)

def entropy_dataset(a_list):  
    # تعديل الدالة لتعرض النصوص باستخدام st.write بدلاً من print
    num_instances = len(a_list) * 1.0
    st.write("\nNumber of Instances of the Current Sub-Class is {0}".format(num_instances))
    
    cnt = Counter(x for x in a_list)
    probs = [x / num_instances for x in cnt.values()]
    st.write("\nClasses➡", "'p'=", max(cnt), "'n'=", min(cnt))
    st.write("\nProbabilities of Class 'p'='{0}' ➡ {1}".format(max(cnt), max(probs)))
    st.write("Probabilities of Class 'n'='{0}'  ➡ {1}".format(min(cnt), min(probs)))
    
    # استدعاء الدالة entropy وإرجاع النتيجة
    return entropy(probs)


def information_gain(df, split_attribute, target_attribute, battr):
    st.write("\n\n----- حساب الربح المعلوماتي للسمة", split_attribute, "-----")

    df_split = df.groupby(split_attribute)
    glist = []
    for gname, group in df_split:
        st.write("قيم السمة المجمعة \n", group)
        st.write("---------------------------------------------------------")
        glist.append(gname)

    glist.reverse()
    nobs = len(df.index) * 1.0
    df_agg1 = df_split.agg(
        {target_attribute: lambda x: entropy_of_list(x, glist.pop())}
    )
    df_agg2 = df_split.agg({target_attribute: lambda x: len(x) / nobs})

    df_agg1.columns = ["الانتروبيا"]
    df_agg2.columns = ["النسبة"]

    new_entropy = sum(df_agg1["الانتروبيا"] * df_agg2["النسبة"])
    if battr != "S":
        old_entropy = entropy_of_list(
            df[target_attribute], "S-" + df.iloc[0][df.columns.get_loc(battr)]
        )
    else:
        old_entropy = entropy_of_list(df[target_attribute], battr)
    return old_entropy - new_entropy


def id3(df, target_attribute, attribute_names, default_class=None, default_attr="S"):

    cnt = Counter(x for x in df[target_attribute])  

    if len(cnt) == 1:
        return next(iter(cnt))

    elif df.empty or (not attribute_names):
        return default_class

    else:
        default_class = max(cnt.keys())
        gainz = []
        for attr in attribute_names:
            ig = information_gain(df, attr, target_attribute, default_attr)
            gainz.append(ig)
            st.write("\nربح المعلومات للسمة", attr, "هو:", ig)
            st.write("=========================================================")

        # index_of_max = gainz.index(max(gainz))
        for s  in gainz:
            index_of_max = gainz.index(max(gainz))

        best_attr = attribute_names[index_of_max]
        st.write(
            "\nقائمة ربح المعلومات للسمات:",
            attribute_names,
            "\nهي:",
            gainz,
            "على التوالي.",
        )
        st.write("\nالسمة ذات أقصى ربح هي:", best_attr)
        st.write("\nبالتالي ، سيكون الجذر هو:", best_attr)
        st.write("=========================================================")

        tree = {best_attr: {}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]


        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(
                data_subset,
                target_attribute,
                remaining_attribute_names,
                default_class,
                best_attr,
            )
            tree[best_attr][attr_val] = subtree
        return tree


def create_graph(tree, graph=None, parent_node=None, parent_edge_label=None):
    if graph is None:
        graph = Digraph(comment="Decision Tree")
    for node, decision_info in tree.items():
        if isinstance(decision_info, dict):
            graph.node(str(node), label=str(node))
            if parent_node is not None:
                graph.edge(str(parent_node), str(node), label=str(parent_edge_label))
            create_graph(decision_info, graph, parent_node=node, parent_edge_label=node)
        else:
            graph.node(str(node), label=str(decision_info))
            if parent_node is not None:
                if parent_edge_label != str(parent_node):
                    graph.edge(str(parent_node), str(node), label=str(parent_edge_label))
                else:
                    graph.edge(str(parent_node), str(node), label="")
    return graph


def classify(instance, tree, default=None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return classify(instance, result)
        else:
            return result
    else:
        return default

def main():
    st.title("Tree Decision ID3 => Upload File CSV")

    # عرض واجهة المستخدم لتحميل الملف
    uploaded_file = st.file_uploader("قم بتحميل ملف CSV", type="csv")
    tab1, tab2,tab3 = st.tabs(
        ["معلومات عامة عن البيانات", "(ID3)شجرة القرار","المخرجات"]
    )
    @st.cache_data
    def load_data(file_name):
        # read CSV file
        data = pd.read_csv(file_name)
        return data

    if uploaded_file is not None:
        # read csv
        df = load_data(uploaded_file)

    with tab1:
        if uploaded_file is not None:
      
            st.header("وصف البيانات")

            row_count = df.shape[0]

            column_count = df.shape[1]

            # Use the duplicated() function to identify duplicate rows
            duplicates = df[df.duplicated()]
            duplicate_row_count = duplicates.shape[0]

            missing_value_row_count = df[df.isna().any(axis=1)].shape[0]

            table_markdown = f"""
        | Description | Value | 
        |---|---|
        | Number of Rows | {row_count} |
        | Number of Columns | {column_count} |
        | Number of Duplicated Rows | {duplicate_row_count} |
        | Number of Rows with Missing Values | {missing_value_row_count} |
        """
            st.markdown(table_markdown)

            st.header("انواع الاعمدة")

            # get feature names
            columns = list(df.columns)

            # create dataframe
            column_info_table = pd.DataFrame(
                {"العمود": columns, "نوع البانات": df.dtypes.tolist()}
            )

            # display pandas dataframe as a table
            st.dataframe(column_info_table, hide_index=True)

        with tab2:
            if uploaded_file is not None:
                # استخراج أسماء الأعمدة
                column_names = df.columns.tolist()
                # اختيار العمود الهدف من المستخدم
                target_column = st.selectbox("اختر العمود الهدف", column_names)

                if target_column:
                    st.write("العمود الهدف هو:", target_column)

                    # استبعاد العمود الهدف من أسماء السمات المتوقعة
                    attribute_names = column_names.copy()
                    attribute_names.remove(target_column)
                    st.write("أسماء السمات المتوقعة:", attribute_names)
                    decision_tree = id3(df, target_column, attribute_names)

                    # if uploaded_file is not None: 
                    df_new = df.copy()
                    # # تطبيق الدالة classify على البيانات وإضافة النتائج إلى عمود "Predicted"

                    # حساب الانتروبي للعمود الهدف
                    entropy_target = entropy_of_list(df[target_column].tolist(), target_column)
                    st.write("الانتروبي للعمود الهدف:", entropy_target)
                    # # عرض البيانات في جدول
                    df_new['Predicted'] = df_new.apply(classify, axis=1, args=(decision_tree, '?'))
                    # st.write(df_new)
                    with tab3:
                        with st.expander("عرض الرسم البياني"):
                            # st.write("Creating Decision Tree Visualization...")
                            graph_source = create_graph(decision_tree)
                            # Set node attributes to display values
                            graph_source.attr("node", shape="box", style="rounded,filled",fontsize="10", fillcolor="lightblue")
                            # Set edge attributes to display split conditions
                            graph_source.attr("edge", fontsize="10" ,color="black")
                            
                            st.graphviz_chart(graph_source)
                            if st.button('Save as PDF'):
                                graph_source.render('decision_tree', format='pdf')
                                st.success('Graph saved as PDF.')
                        with st.expander("عرض شجرة القرار"):
                            st.write(decision_tree)
                        with st.expander("عرض البانات مع التنبؤ"):
                            st.write(df_new)

                




if __name__ == "__main__":
    main()
