import pandas as pd
import openpyxl

# Define a list of lists
# data = [["Aditya", 179],
#         ["Sameer", 181],
#         ["Dharwish", 170],
#         ["Joel", 167]]

# Define column names
#column_names = ["Name", "Height"]

# Define a list of dictionaries
# data = [{"Name": "Aditya", "Height": 179},
#         {"Name": "Sameer", "Height": 181},
#         {"Name": "Dharwish", "Height": 170},
#         {"Name": "Joel", "Height": 167}]

# df = pd.DataFrame(data)
# writer = pd.ExcelWriter('excel_with_list.xlsx', engine='xlsxwriter')

# # Add the pandas dataframe to the excel file as sheet
# df.to_excel(writer, sheet_name='first_sheet', index=False, startrow=3, startcol=3)
# writer.close()


# # Read a csv file into pandas dataframe
# df = pd.read_csv('/Users/zcj/py_workspace/hello/person.csv')
# writer = pd.ExcelWriter('person.xlsx', engine='xlsxwriter')
# df.to_excel(writer, sheet_name='first_sheet', index=False)
# writer.close()


#Define list of dictionaries
# height_data = [{"Name": "Aditya", "Height": 179},
#                {"Name": "Sameer", "Height": 181},
#                {"Name": "Dharwish", "Height": 170},
#                {"Name": "Joel", "Height": 167}]

# weight_data = [{"Name": "Aditya", "Weight": 76},
#                {"Name": "Sameer", "Weight": 68},
#                {"Name": "Dharwish", "Weight": 69},
#                {"Name": "Joel", "Weight": 73}]

# marks_data = [{"Name": "Aditya", "Marks": 79},
#               {"Name": "Sameer", "Marks": 81},
#               {"Name": "Dharwish", "Marks": 70},
#               {"Name": "Joel", "Marks": 67}]

# # Convert list of dictionaries to dataframe
# height_df = pd.DataFrame(height_data)
# weight_df = pd.DataFrame(weight_data)
# marks_df = pd.DataFrame(marks_data)

# writer = pd.ExcelWriter('excel_with_multiple_sheets.xlsx', engine='xlsxwriter')

# height_df.to_excel(writer, sheet_name='height', index=False)
# weight_df.to_excel(writer, sheet_name='weight', index=False)
# marks_df.to_excel(writer, sheet_name='marks', index=False)
# writer.close()

# df = pd.read_excel('excel_with_multiple_sheets.xlsx', sheet_name='marks')
# print("The dataframe is:")
# print(df)

# excel_file = pd.ExcelFile('excel_with_multiple_sheets.xlsx')
# df = pd.read_excel(excel_file, sheet_name="marks")

# print("The dataframe is:")
# print(df)

# df = pd.read_excel('excel_with_multiple_sheets.xlsx',
#                    sheet_name=1, usecols=["Name", "Weight"])
# print("The dataframe column is:")
# print(df)

# # Read a sheet into dataframe directly and extract row
# row = pd.read_excel('excel_with_multiple_sheets.xlsx', sheet_name=1).iloc[2]

# print("The dataframe row is:")
# print(row)

# # read a sheet into dataframe directly and extract the cell
# data = pd.read_excel('excel_with_multiple_sheets.xlsx',
#                      sheet_name=1).iloc[2]["Name"]

# print("The data is:")
# print(data)


# data = pd.read_excel(excel_file, sheet_name=2)["Name"].iloc[2]

# print("The data is:")
# print(data)


# Read existing excel file into ExcelWriter in Append Mode
# writer = pd.ExcelWriter('excel_with_multiple_sheets.xlsx',
#                         mode='a', engine="openpyxl")

# data = [{"Name": "Aditya", "Age": 25},
#         {"Name": "Sameer", "Age": 26},
#         {"Name": "Dharwish", "Age": 24},
#         {"Name": "Joel", "Age": 27}]

# # convert list of dictionaries to dataframe
# df = pd.DataFrame(data)

# # Write the pandas dataframe to the excel file
# df.to_excel(writer, sheet_name='age', index=False)

# # Make sure to properly close the file
# writer.close()

# Read existing excel file into ExcelWriter in Append Mode
# writer = pd.ExcelWriter('excel_with_multiple_sheets.xlsx',
#                         mode='a', engine="openpyxl", if_sheet_exists="replace")
# df = pd.read_excel('excel_with_multiple_sheets.xlsx', sheet_name="weight")
# newRow = {"Name": "Elon", "Weight": 77}
# new_row = pd.DataFrame([newRow])
# df = pd.concat([df, new_row], ignore_index=True)

# # Write the pandas dataframe to the excel file
# df.to_excel(writer, sheet_name='weight', index=False)
# writer.close()


# # Read existing excel file into ExcelWriter in Append Mode
# writer = pd.ExcelWriter('excel_with_multiple_sheets.xlsx',
#                         mode='a', engine="openpyxl", if_sheet_exists="replace")
# df = pd.read_excel('excel_with_multiple_sheets.xlsx', sheet_name="weight")
# df["Weight_lbs"] = df["Weight"]*2.20462
# df["Age"] = [25, 22, 24, 27, 49]

# # Write the pandas dataframe to the excel file
# df.to_excel(writer, sheet_name='weight', index=False)
# writer.close()

# # Read existing excel file into ExcelWriter in Append Mode
# writer = pd.ExcelWriter('excel_with_multiple_sheets.xlsx',
#                         mode='a', engine="openpyxl", if_sheet_exists="new")
# df = pd.read_excel('excel_with_multiple_sheets.xlsx', sheet_name="weight")

# df.to_excel(writer, sheet_name='weight', index=False)
# writer.close()

# spreadsheet = openpyxl.load_workbook('excel_with_multiple_sheets.xlsx')
# ss_sheet = spreadsheet['weight1']
# ss_sheet.title = 'new_weights'
# spreadsheet.save('excel_with_multiple_sheets.xlsx')

# spreadsheet = openpyxl.load_workbook('excel_with_multiple_sheets.xlsx')
# sheet_to_delete = spreadsheet['new_weights']
# spreadsheet.remove(sheet_to_delete)
# spreadsheet.save('excel_with_multiple_sheets.xlsx')

# # Read existing excel file into ExcelWriter in Append Mode
# writer = pd.ExcelWriter('excel_with_multiple_sheets.xlsx',
#                         mode='a', engine="openpyxl", if_sheet_exists="replace")
# df = pd.read_excel('excel_with_multiple_sheets.xlsx', sheet_name="weight")
# df = df.drop([0], axis=0)

# # Write the pandas dataframe to the excel file
# df.to_excel(writer, sheet_name='weight', index=False)
# writer.close()

# # Read existing excel file into ExcelWriter in Append Mode
# writer = pd.ExcelWriter('excel_with_multiple_sheets.xlsx',
#                         mode='a', engine="openpyxl", if_sheet_exists="replace")
# df = pd.read_excel('excel_with_multiple_sheets.xlsx', sheet_name="weight")
# df = df.drop(["Weight_lbs"], axis=1)

# # Write the pandas dataframe to the excel file
# df.to_excel(writer, sheet_name='weight', index=False)
# writer.close()

# Read existing excel file into ExcelWriter in Append Mode
excel_file = pd.ExcelFile('excel_with_multiple_sheets.xlsx')

df1 = pd.read_excel(excel_file, sheet_name="height")
df2 = pd.read_excel(excel_file, sheet_name="weight")
df3 = pd.read_excel(excel_file, sheet_name="marks")

output_df = pd.concat([df1, df2, df3], ignore_index=True)

# Write the pandas dataframe to the csv file
output_df.to_csv("height_data_merged.csv", index=False)