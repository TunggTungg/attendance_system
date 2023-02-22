using ExcelDataReader;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WinFormsApp1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            init();
            dataGridView1.Dock = DockStyle.Fill;
        }


        void init()
        {
            // Add Days
            for (int i = 1; i < 32; i++)
            {
                if (i < 10)
                {
                    day.Items.Add("0" + Convert.ToString(i));
                }
                else
                {
                    day.Items.Add(Convert.ToString(i));
                }
                
            }


            // Add Months
            for (int i = 1; i < 13; i++)
            {
                if (i < 10)
                {
                    month.Items.Add("0" + Convert.ToString(i));
                }
                else
                {
                    month.Items.Add(Convert.ToString(i));
                }
            }

            // Add Years
            for (int i = 22; i < 30; i++)
            {
                if (i < 10)
                {
                    year.Items.Add("0" + Convert.ToString(i));
                }
                else
                {
                    year.Items.Add(Convert.ToString(i));
                }
            }

        }

        DataTableCollection tableCollection;
        string file_path = "Q:/CDT_2_2021/Thi_Giac_May/Project_Giua_Ky/WinFormsApp1/WinFormsApp1/emty.xlsx";
        string dday, mmonth, yyear;
        void show_orders()
        {
            try
            {
                file_path = "Q:/CDT_2_2021/Thi_Giac_May/Project_Giua_Ky/WinFormsApp1/WinFormsApp1/"
                    + dday + "-" + mmonth + "-" + yyear + ".xlsx";

                System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
                using (var stream = File.Open(file_path, FileMode.Open, FileAccess.Read))
                {
                    using (IExcelDataReader reader = ExcelReaderFactory.CreateReader(stream))
                    {
                        DataSet result = reader.AsDataSet(new ExcelDataSetConfiguration()
                        {
                            ConfigureDataTable = (_) => new ExcelDataTableConfiguration() { UseHeaderRow = true }
                        });
                        tableCollection = result.Tables;

                        DataTable dt = tableCollection[0];
                        dataGridView1.DataSource = dt;
                    }
                }
                System.Threading.Thread.Sleep(500);
            }
            catch (Exception ex)
            {
                file_path = "Q:/CDT_2_2021/Thi_Giac_May/Project_Giua_Ky/WinFormsApp1/WinFormsApp1/emty.xlsx";
            }
        }
        private void timer1_Tick(object sender, EventArgs e)
        {
            show_orders();
        }

        private void day_SelectedIndexChanged(object sender, EventArgs e)
        {
            dday = day.Text;
        }

        private void month_SelectedIndexChanged(object sender, EventArgs e)
        {
            mmonth = month.Text;
        }

        private void year_SelectedIndexChanged(object sender, EventArgs e)
        {
            yyear = year.Text;
        }
    }
}

