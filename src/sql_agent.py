"""
SQL Agent Module for Natural Language to SQL Conversion
Simple, reliable SQL agent with proper column quoting
"""

import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
import warnings
import random
from datetime import datetime
warnings.filterwarnings('ignore')

class SQLAgent:
    """Simple SQL agent for natural language queries with robust error handling"""
    
    def __init__(self, config=None):
        self.config = config
        self.database_path = "data/autonomous_agent.db"
        self.query_history = []
        self.table_schemas = {}
        
        # Create database directory
        Path("data").mkdir(exist_ok=True)
        
        print("SQL Agent initialized")
    
    def setup_database(self, df: pd.DataFrame, table_name: str = "main_table"):
        """Setup SQLite database with the dataframe"""
        
        try:
            # Create connection
            conn = sqlite3.connect(self.database_path)
            
            # Store dataframe as SQL table
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Store schema information
            self.table_schemas[table_name] = {
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample_data': df.head(3).to_dict('list'),
                'row_count': len(df)
            }
            
            conn.close()
            print(f"✅ Database setup completed - Table: {table_name}")
            
        except Exception as e:
            print(f"❌ Error setting up database: {str(e)}")
            raise
    
    def natural_language_to_sql(self, user_query: str, df: pd.DataFrame, table_name: str = "main_table") -> str:
        """Convert natural language to SQL query with proper column quoting"""
        try:
            print(f"Converting query: {user_query}")
            
            # Setup database if not already done
            if table_name not in self.table_schemas:
                self.setup_database(df, table_name)
            
            # Use simplified rule-based approach
            sql_query = self._simple_nlp_to_sql(user_query, table_name)
            
            print(f"✅ Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            print(f"❌ Error converting query: {str(e)}")
            return f'SELECT * FROM "{table_name}" LIMIT 10;'
    
    def _quote_column(self, column_name: str) -> str:
        """Properly quote column names to handle spaces and special characters"""
        # Remove any existing quotes first
        column_name = column_name.replace('"', '').replace('`', '').replace("'", "")
        return f'"{column_name}"'
    
    def _simple_nlp_to_sql(self, user_query: str, table_name: str) -> str:
        """Simplified natural language to SQL conversion"""
        
        try:
            query_lower = user_query.lower()
            schema_info = self.table_schemas[table_name]
            columns = schema_info['columns']
            
            # Always quote table name
            quoted_table = f'"{table_name}"'
            
            # BASIC QUERY PATTERNS - Simple and reliable
            if any(word in query_lower for word in ['show me all', 'display all', 'all data', 'everything']):
                return f'SELECT * FROM {quoted_table} LIMIT 100;'
            
            elif any(word in query_lower for word in ['count', 'number of', 'how many']):
                return f'SELECT COUNT(*) as count FROM {quoted_table};'
            
            elif any(word in query_lower for word in ['first', 'top', 'sample']):
                limit_match = re.search(r'(\d+)', user_query)
                limit_num = limit_match.group(1) if limit_match else "10"
                return f'SELECT * FROM {quoted_table} LIMIT {limit_num};'
            
            elif any(word in query_lower for word in ['last', 'bottom']):
                return f'SELECT * FROM {quoted_table} ORDER BY rowid DESC LIMIT 10;'
            
            elif any(word in query_lower for word in ['missing', 'null', 'empty']):
                # Check which columns have null values
                null_checks = []
                for col in columns:
                    quoted_col = self._quote_column(col)
                    null_checks.append(f"{quoted_col} IS NULL")
                where_clause = " OR ".join(null_checks[:3])  # Limit to first 3 columns
                return f'SELECT * FROM {quoted_table} WHERE {where_clause} LIMIT 50;'
            
            # COLUMN-SPECIFIC QUERIES
            mentioned_columns = []
            for col in columns:
                col_lower = col.lower()
                if any(word in col_lower for word in query_lower.split()):
                    mentioned_columns.append(col)
            
            if mentioned_columns:
                # Quote all mentioned columns
                quoted_columns = [self._quote_column(col) for col in mentioned_columns]
                select_clause = ", ".join(quoted_columns)
                
                # Add basic aggregations if requested
                if any(word in query_lower for word in ['average', 'avg', 'mean']):
                    numeric_cols = [col for col in mentioned_columns 
                                  if any(dtype in schema_info['dtypes'][col] 
                                       for dtype in ['int', 'float', 'number'])]
                    if numeric_cols:
                        return f'SELECT AVG({self._quote_column(numeric_cols[0])}) as average FROM {quoted_table};'
                
                elif any(word in query_lower for word in ['sum', 'total']):
                    numeric_cols = [col for col in mentioned_columns 
                                  if any(dtype in schema_info['dtypes'][col] 
                                       for dtype in ['int', 'float', 'number'])]
                    if numeric_cols:
                        return f'SELECT SUM({self._quote_column(numeric_cols[0])}) as total FROM {quoted_table};'
                
                elif any(word in query_lower for word in ['max', 'maximum', 'highest']):
                    numeric_cols = [col for col in mentioned_columns 
                                  if any(dtype in schema_info['dtypes'][col] 
                                       for dtype in ['int', 'float', 'number'])]
                    if numeric_cols:
                        return f'SELECT MAX({self._quote_column(numeric_cols[0])}) as maximum FROM {quoted_table};'
                
                elif any(word in query_lower for word in ['min', 'minimum', 'lowest']):
                    numeric_cols = [col for col in mentioned_columns 
                                  if any(dtype in schema_info['dtypes'][col] 
                                       for dtype in ['int', 'float', 'number'])]
                    if numeric_cols:
                        return f'SELECT MIN({self._quote_column(numeric_cols[0])}) as minimum FROM {quoted_table};'
                
                # Simple select with mentioned columns
                return f'SELECT {select_clause} FROM {quoted_table} LIMIT 10;'
            
            # DEFAULT: Return limited data with all columns
            return f'SELECT * FROM {quoted_table} LIMIT 10;'
            
        except Exception as e:
            print(f"❌ Error in simple NLP-to-SQL: {str(e)}")
            return f'SELECT * FROM "{table_name}" LIMIT 10;'
    
    def execute_sql_query(self, sql_query: str, max_rows: int = 1000) -> pd.DataFrame:
        """Execute SQL query and return results with robust error handling"""
        
        try:
            print(f"Executing SQL: {sql_query}")
            
            # Connect to database
            conn = sqlite3.connect(self.database_path)
            
            # Execute query
            result_df = pd.read_sql_query(sql_query, conn)
            
            # Limit results
            if len(result_df) > max_rows:
                result_df = result_df.head(max_rows)
                print(f"⚠️ Results limited to {max_rows} rows")
            
            conn.close()
            
            # Store in query history
            self.query_history.append({
                'query': sql_query,
                'timestamp': pd.Timestamp.now(),
                'result_count': len(result_df),
                'columns': result_df.columns.tolist()
            })
            
            print(f"✅ Query executed successfully - {len(result_df)} rows returned")
            return result_df
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error executing SQL query: {error_msg}")
            
            # Try to auto-correct common issues
            corrected_sql = self._auto_fix_sql(sql_query, error_msg)
            if corrected_sql != sql_query:
                print(f"🔄 Attempting auto-corrected SQL: {corrected_sql}")
                try:
                    conn = sqlite3.connect(self.database_path)
                    result_df = pd.read_sql_query(corrected_sql, conn)
                    conn.close()
                    print(f"✅ Auto-corrected query executed successfully")
                    return result_df
                except Exception as e2:
                    print(f"❌ Auto-correct also failed: {str(e2)}")
            
            # Return empty dataframe with error info
            return pd.DataFrame({'Error': [error_msg]})
    
    def _auto_fix_sql(self, sql_query: str, error_msg: str) -> str:
        """Auto-fix common SQL errors"""
        
        # Fix unquoted column names with spaces
        if "no such column" in error_msg.lower():
            # Extract the problematic column name from error
            column_match = re.search(r"no such column: ([^\s]+)", error_msg.lower())
            if column_match:
                wrong_column = column_match.group(1)
                # Quote the column name in the SQL
                corrected_sql = sql_query.replace(wrong_column, f'"{wrong_column}"')
                return corrected_sql
        
        # Fix table name issues
        if "no such table" in error_msg.lower():
            # Ensure table is quoted
            if "main_table" in sql_query and '"main_table"' not in sql_query:
                return sql_query.replace("main_table", '"main_table"')
        
        return sql_query
    
    def suggest_visualization(self, result_df: pd.DataFrame, user_query: str = "") -> Optional[go.Figure]:
        """Suggest appropriate visualization for query results"""
        
        try:
            if result_df.empty or 'Error' in result_df.columns:
                return None
            
            print("🤖 Creating visualization...")
            
            # Analyze result structure
            n_rows, n_cols = result_df.shape
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Simple visualization logic
            if n_cols == 1:
                col = result_df.columns[0]
                if col in numeric_cols:
                    return px.histogram(result_df, x=col, title=f"Distribution of {col}")
                else:
                    value_counts = result_df[col].value_counts().head(10)
                    return px.bar(x=value_counts.index, y=value_counts.values, 
                                 title=f"Frequency of {col}")
            
            elif n_cols >= 2:
                x_col, y_col = result_df.columns[0], result_df.columns[1]
                
                if x_col in categorical_cols and y_col in numeric_cols:
                    return px.bar(result_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                elif x_col in numeric_cols and y_col in numeric_cols:
                    return px.scatter(result_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                else:
                    return px.bar(result_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error creating visualization: {str(e)}")
            return None
    
    def create_bar_chart(self, result_df: pd.DataFrame) -> Optional[go.Figure]:
        """Create simple bar chart"""
        try:
            if result_df.empty or len(result_df.columns) < 2:
                return None
                
            x_col, y_col = result_df.columns[0], result_df.columns[1]
            return px.bar(result_df, x=x_col, y=y_col, title=f"Bar Chart: {y_col} by {x_col}")
            
        except Exception as e:
            print(f"❌ Error creating bar chart: {str(e)}")
            return None

    def create_line_chart(self, result_df: pd.DataFrame) -> Optional[go.Figure]:
        """Create simple line chart"""
        try:
            if result_df.empty or len(result_df.columns) < 2:
                return None
                
            x_col, y_col = result_df.columns[0], result_df.columns[1]
            return px.line(result_df, x=x_col, y=y_col, title=f"Line Chart: {y_col} over {x_col}")
            
        except Exception as e:
            print(f"❌ Error creating line chart: {str(e)}")
            return None

    def create_pie_chart(self, result_df: pd.DataFrame) -> Optional[go.Figure]:
        """Create simple pie chart"""
        try:
            if result_df.empty:
                return None
                
            if len(result_df.columns) >= 2:
                cat_col, val_col = result_df.columns[0], result_df.columns[1]
                return px.pie(result_df, names=cat_col, values=val_col, title=f"Pie Chart: {val_col} by {cat_col}")
            else:
                cat_col = result_df.columns[0]
                value_counts = result_df[cat_col].value_counts().head(10)
                return px.pie(values=value_counts.values, names=value_counts.index, 
                             title=f"Distribution of {cat_col}")
                
        except Exception as e:
            print(f"❌ Error creating pie chart: {str(e)}")
            return None

    def create_scatter_plot(self, result_df: pd.DataFrame) -> Optional[go.Figure]:
        """Create simple scatter plot"""
        try:
            if result_df.empty or len(result_df.columns) < 2:
                return None
                
            x_col, y_col = result_df.columns[0], result_df.columns[1]
            return px.scatter(result_df, x=x_col, y=y_col, title=f"Scatter Plot: {y_col} vs {x_col}")
            
        except Exception as e:
            print(f"❌ Error creating scatter plot: {str(e)}")
            return None

    def create_heatmap(self, result_df: pd.DataFrame) -> Optional[go.Figure]:
        """Create simple heatmap"""
        try:
            if result_df.empty:
                return None
                
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                corr_matrix = result_df[numeric_cols].corr()
                return px.imshow(corr_matrix, title="Correlation Heatmap", aspect="auto")
            return None
            
        except Exception as e:
            print(f"❌ Error creating heatmap: {str(e)}")
            return None
    
    def get_query_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate reliable query suggestions - always returns exactly 9 queries"""
        
        try:
            columns = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Always start with these 9 reliable queries
            suggestions = [
                "Show me all data",
                "Count all records",  
                "Show me first 10 rows",
                "What are the column names?",
                "Give me data summary",
                "Find missing values",
                "Show me the last 5 records",
                "Display random sample of data",
                "Show data structure info"
            ]
            
            # If we have specific columns, enhance suggestions
            if numeric_cols:
                col = numeric_cols[0]
                enhanced_queries = [
                    f"Show average of {col}",
                    f"Find maximum {col}",
                    f"Calculate total {col}",
                    f"Show {col} statistics",
                    f"Find top 10 by {col}",
                    f"Show distribution of {col}"
                ]
                # Replace some generic queries with enhanced ones
                suggestions[4:6] = enhanced_queries[:2]  # Replace positions 4-5
            
            if categorical_cols:
                col = categorical_cols[0]
                cat_queries = [
                    f"Count records by {col}",
                    f"Show unique values in {col}",
                    f"Most frequent {col} values",
                    f"Group data by {col}"
                ]
                # Replace positions 6-7 with categorical queries
                suggestions[6:8] = cat_queries[:2]
            
            # Ensure we always have exactly 9 queries
            return suggestions[:9]
            
        except Exception as e:
            print(f"❌ Error generating query suggestions: {str(e)}")
            # Fallback - always return 9 queries
            return [
                "Show me all data",
                "Count all records", 
                "Show me first 10 rows",
                "What are the column names?",
                "Give me data summary",
                "Find missing values",
                "Show me the last 5 records",
                "Display random sample",
                "Show data structure"
            ]
    
    def auto_correct_sql(self, sql_query: str, error_message: str, table_name: str) -> str:
        """Simple auto-correction for SQL queries"""
        
        try:
            print(f"Auto-correcting SQL query...")
            
            # Quote column names with spaces
            if "no such column" in error_message.lower():
                # Simple approach: quote all potential column names in the SELECT clause
                if "SELECT" in sql_query.upper() and "FROM" in sql_query.upper():
                    select_part = sql_query.upper().split("FROM")[0]
                    # Add quotes around column names that might have spaces
                    corrected = sql_query
                    for word in select_part.split():
                        if word not in ['SELECT', '*', ',', ''] and not word.isupper():
                            if ' ' in word or '(' not in word:
                                corrected = corrected.replace(word, f'"{word}"')
                    return corrected
            
            # Fallback to simple query
            return f'SELECT * FROM "{table_name}" LIMIT 10;'
            
        except Exception as e:
            print(f"❌ Error auto-correcting SQL: {str(e)}")
            return f'SELECT * FROM "{table_name}" LIMIT 10;'
    
    def get_query_explanation(self, sql_query: str) -> str:
        """Generate simple query explanation"""
        
        try:
            if "COUNT" in sql_query.upper():
                return "This query counts the total number of records in the dataset."
            elif "SELECT *" in sql_query.upper():
                return "This query displays all data from the dataset with some limits."
            elif "AVG" in sql_query.upper() or "AVERAGE" in sql_query.upper():
                return "This query calculates the average value of a numeric column."
            elif "SUM" in sql_query.upper():
                return "This query calculates the total sum of a numeric column."
            elif "MAX" in sql_query.upper():
                return "This query finds the maximum value in a column."
            elif "MIN" in sql_query.upper():
                return "This query finds the minimum value in a column."
            else:
                return "This query retrieves specific data based on your request."
                
        except Exception as e:
            return "This query retrieves data from the dataset."
    
    def analyze_query_complexity(self, sql_query: str) -> str:
        """Simple complexity analysis"""
        try:
            query_upper = sql_query.upper()
            
            if any(word in query_upper for word in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                return "Simple"
            elif any(word in query_upper for word in ['WHERE', 'GROUP BY', 'ORDER BY']):
                return "Moderate"
            else:
                return "Simple"
        except:
            return "Simple"