import pathway as pw

input_table = pw.demo.range_stream()

sum_table = input_table.reduce(sum= pw.reducers.sum(input_table.value))

pw.io.csv.write(sum_table, 'output_stream.csv')

pw.run()
