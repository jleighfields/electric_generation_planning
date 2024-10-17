import sqlite3
import pandas as pd
import os
import shutil

# set up logging
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ResultsDB:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.table_names = ['inputs', 'cap_mw', 'metrics', 'final_df']


    def clear_db(self):
        try:
            cur = self.db.cursor()
            sql_text = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = cur.execute(sql_text).fetchall()
            tables = [t[0] for t in tables]
            for table_name in self.table_names:
                if table_name in tables:
                    sql_text = f"DELETE from {table_name}"
                    log.info(f"sql_text: {sql_text}")
                    cur.execute(sql_text)

        except Exception as e:
            log.error('Error thrown during delete_run')
            log.error(e)

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
                log.info(f'deleting old run {results["run_name"]}')
                self.delete_run(results["run_name"])

            for table_name in self.table_names:
                log.info(f"adding {table_name} for run {results['run_name']}")
                if table_name != 'final_df':
                    df = pd.DataFrame(results[table_name], index=[results['run_name']])
                    df.index.names = ['run_name']
                else:
                    df = results[table_name].copy()
                    df['run_name'] = results['run_name']
                    df.set_index('run_name', inplace=True)
                df.to_sql(name=table_name, con=self.db, if_exists='append', index=True)

        except Exception as e:
            log.error('Error thrown during add run data')
            log.error(e)



    def delete_run(self, run):
        try:
            cur = self.db.cursor()
            for table_name in self.table_names:
                sql_text = f"DELETE from {table_name} WHERE run_name = ?"
                cur.execute(sql_text, (run,))

        except Exception as e:
            log.error('Error thrown during delete_run')
            log.error(e)

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
            res = temp_df.run_name.unique()
        else:
            res =  []

        if cur:
            cur.close()

        return res


    def zip_results(self):
        log.info('zipping results')
        zip_loc = './csv'
        if os.path.exists(zip_loc):
            shutil.rmtree(zip_loc)
        os.mkdir(zip_loc)

        try:
            cur = self.db.cursor()
            sql_text = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = cur.execute(sql_text).fetchall()
            tables = [t[0] for t in tables]
            for table_name in self.table_names:
                if table_name in tables:
                    sql_text = f"SELECT * from {table_name}"
                    save_path = f'{zip_loc}/{table_name}.csv'
                    df = pd.read_sql_query(sql_text, self.db)
                    df.to_csv(save_path)
                    log.info(f'zipping table: {table_name}')
                    log.info(f'n_rows: {df.shape[0]}')

            shutil.make_archive('results', 'zip', zip_loc)

        except Exception as e:
            log.error('Error thrown during zipping results')
            log.error(e)
        finally:
            if cur:
                cur.close()

    def print_head(self):
        for table_name in self.table_names:
            sql_text = f"SELECT * from {table_name}"
            temp_df = pd.read_sql_query(sql_text, self.db)
            log.info(f'table: {table_name}')
            log.info(temp_df.head())
            print('\n')


if __name__ == '__main__':
    print('\n')

    ##############################################
    # get results dictionary for testing

    from joblib import load

    # if not os.path.isfile('results.joblib'):

    # run LP.py to get joblib files
    results = load('results.joblib')
    results2 = load('results2.joblib')

    print('\n################################')
    log.info('ADD test run')
    test_db = ResultsDB()
    log.info(type(test_db).__name__)
    test_db.clear_db()
    test_db.add_run(results)
    test_db.add_run(results2)
    test_db.print_head()

    print('\n################################')
    log.info('GET runs')
    log.info(f'len: {len(test_db.get_runs())}')
    log.info(test_db.get_runs())
    assert len(test_db.get_runs()) == 2

    print('\n################################')
    log.info('DELETE test')
    test_db.delete_run('test')
    log.info(test_db.get_runs())
    assert len(test_db.get_runs()) == 1
    test_db.print_head()

    print('\n################################')
    log.info('ZIP test')
    test_db.zip_results()
    assert os.path.isfile('results.zip')

    print('\n################################')
    log.info('CLEAR DB test')
    test_db.clear_db()
    log.info(test_db.get_runs())
    assert len(test_db.get_runs()) == 0




