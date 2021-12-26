import datetime
import sqlite3
import pandas as pd
import os
import shutil

class ResultsDB:

    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.table_names = ['inputs', 'cap_mw', 'metrics', 'final_df']



    def clear_db(self):
        conn = None
        try:
            cur = self.db.cursor()
            sql_text = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = cur.execute(sql_text).fetchall()
            tables = [t[0] for t in tables]
            print(f'tables: {tables}')
            for table_name in self.table_names:
                if table_name in tables:
                    sql_text = "DELETE from {table}".format(**{'table': table_name})
                    cur.execute(sql_text)

            cur.close()

        except Exception as e:
            print('Error thrown during delete_run')
            print(e)
        finally:
            if cur:
                cur.close()


    def add_run(self, results):
        '''
        input: results dictionary from LP_ortools_func.run_lp()
            {'run_name': run_name,
            'inputs': inputs,
            'obj_val': obj_val,
            'cap_mw': cap_mw,
            'metrics': metrics,
            'final_df': final_df}

        data to save:
            inputs
            cap_mw
            metrics
            final_df
        '''

        try:
            if results['run_name'] in self.get_runs():
                print(f'deleting old run {results["run_name"]}')
                self.delete_run(results["run_name"])

            for table_name in self.table_names:
                print(f"adding {table_name} for run {results['run_name']}")
                if table_name != 'final_df':
                    df = pd.DataFrame(results[table_name], index=[results['run_name']])
                    df.index.names = ['run_name']
                else:
                    df = results[table_name].copy()
                    df['run_name'] = results['run_name']
                    df.set_index('run_name', inplace=True)
                df.to_sql(name=table_name, con=self.db, if_exists='append', index=True)

        except Exception as e:
            print('Error thrown during add run data')
            print(e)



    def delete_run(self, run):
        try:
            cur = self.db.cursor()
            for table_name in self.table_names:
                sql_text = "DELETE from {table} WHERE run_name = ?".format(**{'table': table_name})
                cur.execute(sql_text, (run,))

            cur.close()

        except Exception as e:
            print('Error thrown during delete_run')
            print(e)
        finally:
            if cur:
                cur.close()


    def get_runs(self):
        cur = self.db.cursor()
        sql_text = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = cur.execute(sql_text).fetchall()
        tables = [t[0] for t in tables]
        if 'inputs' in tables:
            sql_text = "select run_name from inputs"
            temp_df = pd.read_sql_query(sql_text, self.db)
            return temp_df.run_name.unique()
        else:
            return []


    def zip_results(self):
        print('zipping results')
        zip_loc = './csv'
        if os.path.exists(zip_loc):
            shutil.rmtree(zip_loc)
        os.mkdir(zip_loc)

        try:
            cur = self.db.cursor()
            sql_text = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = cur.execute(sql_text).fetchall()
            tables = [t[0] for t in tables]
            print(f'tables: {tables}')
            for table_name in self.table_names:
                if table_name in tables:
                    sql_text = "SELECT * from {table}".format(**{'table': table_name})
                    save_path = f'{zip_loc}/{table_name}.csv'
                    pd.read_sql_query(sql_text, self.db).to_csv(save_path)

            shutil.make_archive('results', 'zip', zip_loc)

        except Exception as e:
            print('Error thrown during delete_run')
            print(e)
        finally:
            if cur:
                cur.close()

    def print_head(self):
        for table_name in self.table_names:
            sql_text = "select * from {table}".format(**{'table': table_name})
            temp_df = pd.read_sql_query(sql_text, self.db)
            print(f'table: {table_name}')
            print(temp_df.head())
            print('\n')


if __name__ == '__main__':
    print('\n')

    ##############################################
    # get results dictionary for testing

    import datetime
    from joblib import load

    # run LP_ortools_func.py to get joblib files
    results = load('results.joblib')
    results2 = load('results2.joblib')

    # table_names = ['inputs', 'cap_mw', 'metrics', 'final_df']
    # print(pd.DataFrame(results[table_names[0]], index=[results['run_name']]))

    print('\n################################')
    print('ADD test run')
    test_db = ResultsDB()
    test_db.clear_db()
    test_db.add_run(results)
    test_db.add_run(results2)
    test_db.print_head()

    print('\n################################')
    print('GET runs')
    print(f'len: {len(test_db.get_runs())}')
    print(test_db.get_runs())

    print('\n################################')
    print('DELETE test')
    test_db.delete_run('test')
    test_db.print_head()

    print('\n################################')
    print('ZIP test')
    test_db.zip_results()


